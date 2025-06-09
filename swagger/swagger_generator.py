import ast
import importlib.util
import inspect
import json
import os
import glob
from typing import get_origin, get_args, List, Optional, Dict, Any, Union
import enum

import frappe
from pydantic import BaseModel



def _extract_name_from_node(node: ast.AST) -> Optional[str]:
    """
    Recursively reconstructs a dotted name from ast.Name or ast.Attribute.
    Returns None for any other node type.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _extract_name_from_node(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None

def find_pydantic_model_in_decorator(node: ast.AST) -> Optional[str]:
    """
    Look for @validate_request(MyModel) or @rest_handler([...], MyModel)
    or @rest_handler([...], model=MyModel). Returns the model name string
    if found, else None.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef):
            for decorator in n.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue

                # Identify decorator by name
                func = decorator.func
                if not (isinstance(func, ast.Name) and func.id in ("validate_request", "rest_handler")):
                    continue

                # Case 1: @validate_request(MyModel)
                if func.id == "validate_request":
                    if decorator.args:
                        candidate = decorator.args[0]
                        name = _extract_name_from_node(candidate)
                        if name:
                            return name

                # Case 2: @rest_handler([...], MyModel) or @rest_handler([...], model=MyModel)
                elif func.id == "rest_handler":
                    # positional: methods list is args[0], model would be args[1]
                    if len(decorator.args) > 1:
                        candidate = decorator.args[1]
                        name = _extract_name_from_node(candidate)
                        if name:
                            return name

                    # keyword: look for keyword arg named "model"
                    for kw in decorator.keywords:
                        if kw.arg == "model":
                            name = _extract_name_from_node(kw.value)
                            if name:
                                return name

    return None

def get_pydantic_model_schema(model_name, module):
    """Extract the schema from a Pydantic model.
    
    Args:
        model_name (str): The name of the Pydantic model.
        module (module): The module where the model is defined.
    
    Returns:
        dict: The JSON schema of the Pydantic model, if valid.
    """
    if hasattr(module, model_name):
        model = getattr(module, model_name)
        if issubclass(model, BaseModel):
            return model.model_json_schema()
    return None

def is_allow_guest_in_whitelist(func):
    """Check if @frappe.whitelist(allow_guest=True) is present on the function."""
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if (
                            hasattr(decorator.func, "id")
                            and decorator.func.id == "frappe"
                            and decorator.attr == "whitelist"
                        ):
                            for kw in decorator.keywords:
                                if kw.arg == "allow_guest" and getattr(kw.value, "value", False):
                                    return True
                    elif (
                        isinstance(decorator, ast.Attribute)
                        and decorator.attr == "whitelist"
                    ):
                        # No allow_guest specified, default is False
                        return False
        return False
    except Exception:
        return False

def process_function(
    app_name: str,
    module_name: str,
    func_name: str,
    func: Any,
    swagger: Dict[str, Any],
    module: Any,
    rel_path_dotted: str = None,
    tag: str = None,
):
    try:
        source = inspect.getsource(func)
        tree   = ast.parse(source)

        # Skip non‐API functions
        if not any(
            (
                ("validate_http_method" in ast.dump(node) or "rest_handler" in ast.dump(node))
                and isinstance(node, ast.Call)
            )
            for node in ast.walk(tree)
        ):
            return

        # Ensure components & schemas containers exist
        components = swagger.setdefault("components", {})
        schemas    = components.setdefault("schemas", {})

        # Recursive flattener for JSON‐Schema dicts
        def extract_and_register_defs(schema_dict: Dict[str,Any]):
            """
            If schema_dict contains '$defs' or 'definitions', pull them out,
            register each nested schema, and recurse.
            """
            for key in ("$defs", "definitions"):
                if key in schema_dict:
                    nested = schema_dict.pop(key)
                    for subname, subschema in nested.items():
                        # Recurse into that subschema first
                        extract_and_register_defs(subschema)
                        # Then register it
                        schemas[subname] = subschema

        # Register a BaseModel or Enum (and all their nested definitions)
        def register_schema(cls: type) -> None:
            name = cls.__name__
            if name in schemas:
                return
            if inspect.isclass(cls) and issubclass(cls, BaseModel):
                # Let Pydantic build the root schema (which may include nested $defs)
                root = cls.schema(ref_template="#/components/schemas/{model}")
                # Flatten any nested definitions
                extract_and_register_defs(root)
                # Finally register the root
                schemas[name] = root
            elif inspect.isclass(cls) and issubclass(cls, enum.Enum):
                schemas[name] = {
                    "type": "string",
                    "enum": [member.value for member in cls],
                }

        # --- Build path & verb ---
        if rel_path_dotted:
            path = f"/api/method/{app_name}.{rel_path_dotted}.{func_name}".lower()
        else:
            path = f"/api/method/{app_name}.api.{module_name}.{func_name}".lower()

        http_method = "post"
        for m in ("get","post","put","delete","patch","options","head"):
            if m.upper() in source:
                http_method = m
                break

        parameters: List[Dict[str,Any]] = []
        request_body: Dict[str,Any] = {}

        sig = inspect.signature(func)
        # --- Params & Request Body ---
        for pname, param in sig.parameters.items():
            if pname in ("self","cls"):
                continue

            ann     = param.annotation
            default = param.default

            # Body: a Pydantic model
            if (
                inspect.isclass(ann)
                and issubclass(ann, BaseModel)
                and http_method in ("post","put","patch")
            ):
                register_schema(ann)
                request_body = {
                    "description": (ann.__doc__ or "Request body").strip(),
                    "required": default is inspect._empty,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": f"#/components/schemas/{ann.__name__}"}
                        }
                    },
                }
                continue

            # Query param: primitive, Enum, or List[...]
            schema = {"type": "string"}
            if inspect.isclass(ann) and issubclass(ann, enum.Enum):
                register_schema(ann)
                schema = {"$ref": f"#/components/schemas/{ann.__name__}"}
            else:
                origin = get_origin(ann)
                args   = get_args(ann)
                if origin in (list,List) and args:
                    item = args[0]
                    if inspect.isclass(item) and issubclass(item, BaseModel):
                        register_schema(item)
                        item_schema = {"$ref": f"#/components/schemas/{item.__name__}"}
                    elif inspect.isclass(item) and issubclass(item, enum.Enum):
                        register_schema(item)
                        item_schema = {"$ref": f"#/components/schemas/{item.__name__}"}
                    else:
                        t = item if item in (int,float,bool,str) else str
                        item_schema = {
                            "type": (
                                "integer" if t is int else
                                "number"  if t is float else
                                "boolean" if t is bool else
                                "string"
                            )
                        }
                    schema = {"type": "array", "items": item_schema}
                elif ann in (int,float,bool,str):
                    schema = {
                        "type": (
                            "integer" if ann is int else
                            "number"  if ann is float else
                            "boolean" if ann is bool else
                            "string"
                        )
                    }

            parameters.append({
                "name":        pname,
                "in":          "query",
                "required":    default is inspect._empty,
                "schema":      schema,
                "description": "",
            })

        # --- Responses ---
        responses = {
            "200": {
                "description": "Successful response",
                "content": {"application/json": {"schema": {"type": "object"}}}
            }
        }
        ret = sig.return_annotation
        if inspect.isclass(ret) and issubclass(ret, BaseModel):
            register_schema(ret)
            responses["200"]["content"]["application/json"]["schema"] = {
                "$ref": f"#/components/schemas/{ret.__name__}"
            }
        elif inspect.isclass(ret) and issubclass(ret, enum.Enum):
            register_schema(ret)
            responses["200"]["content"]["application/json"]["schema"] = {
                "$ref": f"#/components/schemas/{ret.__name__}"
            }

        # --- Guest vs Auth ---
        allow_guest = False
        for deco in tree.body[0].decorator_list:
            if isinstance(deco, ast.Call):
                fn = deco.func
                if (
                    (isinstance(fn, ast.Attribute) and fn.attr == "whitelist")
                    or (isinstance(fn, ast.Name)      and fn.id   == "frappe.whitelist")
                ):
                    for kw in deco.keywords:
                        if kw.arg == "allow_guest" and getattr(kw.value, "value", False):
                            allow_guest = True

        # --- Assemble operation object ---
        operation: Dict[str,Any] = {
            "summary":     func_name.replace("_"," ").title(),
            "description": inspect.getdoc(func) or "",
            "tags":        [tag or module_name],
            "parameters":  parameters,
            "responses":   responses,
            **({"requestBody": request_body} if request_body else {}),
            "security":    [] if allow_guest else [{"basicAuth": []}],
        }

        # --- Inject into swagger.paths ---
        swagger.setdefault("paths", {}).setdefault(path, {})[http_method] = operation

    except Exception as e:
        frappe.log_error(f"Error in process_function for {func_name}: {e}")

def load_module_from_file(file_path):
    """Load a module dynamically from a given file path.
    
    Args:
        file_path (str): The file path of the module.
    
    Returns:
        module: The loaded module.
    """
    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def find_all_files_with_ext(root_dir,ext):
    pattern = os.path.join(root_dir, "**", f"*.{ext}")
    return glob.glob(pattern, recursive=True)

@frappe.whitelist(allow_guest=True)
def generate_swagger_json(*args, **kwargs):
    """Generate Swagger JSON documentation for all API methods.
    
    This function processes all Python files in the `api` directories of installed apps
    to generate a Swagger JSON file that describes the API methods.
    """
    print("[Swagger] generate_swagger_json called")
    swagger_settings = frappe.get_single("Swagger Settings")
    
    # Get the current site URL
    site_url = frappe.utils.get_url()
    
    # Initialize the Swagger specification
    swagger = {
        "openapi": "3.0.0",
        "info": {
            "title": f"{swagger_settings.app_name} API",
            "version": "1.0.0",
        },
        "servers": [
            {
                "url": site_url,
                "description": "Current site server"
            }
        ],
        "paths": {},
        "components": {},
    }

    # Add security schemes based on the settings in "Swagger Settings"
    if swagger_settings.token_based_basicauth or swagger_settings.bearerauth:
        swagger["components"]["securitySchemes"] = {}
        swagger["security"] = []

    if swagger_settings.token_based_basicauth:
        swagger["components"]["securitySchemes"]["basicAuth"] = {
            "type": "http",
            "scheme": "basic",
        }
        swagger["security"].append({"basicAuth": []})

    if swagger_settings.bearerauth:
        swagger["components"]["securitySchemes"]["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
        swagger["security"].append({"bearerAuth": []})

    # Get the path to the Frappe bench directory
    frappe_bench_dir = frappe.utils.get_bench_path()
    file_paths = []

    # Hardcoded list of possible endpoint folders (relative to bench dir)
    for app in frappe.get_installed_apps():
        app_name = app
        app_main_folder = os.path.join(frappe_bench_dir, f"apps/{app_name}/{app_name}")
        endpoint_folders = [
            (f"apps/{app_name}/{app_name}/{app_name}/core/endpoints/", "Core - "),
            (f"apps/{app_name}/{app_name}/{app_name}/endpoints/", "Application - "),
        ]
        for endpoint_folder in endpoint_folders:
            folder, tag_prefix = endpoint_folder
            abs_folder = os.path.join(frappe_bench_dir, folder)
            if os.path.exists(abs_folder) and os.path.isdir(abs_folder):
                py_files = find_all_files_with_ext(abs_folder, "py")
                for file_path in py_files:
                    # Compute the relative path from the app's main folder (excluding .py)
                    rel_path = os.path.relpath(file_path, app_main_folder)
                    rel_path_no_ext, _ = os.path.splitext(rel_path)
                    rel_path_dotted = rel_path_no_ext.replace(os.sep, ".")

                    # Generate the tags
                    tag = f"{tag_prefix}{os.path.splitext(os.path.relpath(file_path, abs_folder))[0].replace('_', ' ').title()}"
                    file_paths.append((app_name, file_path, rel_path_dotted, tag))

    # Process each Python file found
    for app, file_path, rel_path_dotted, tag in file_paths:
        try:
            if os.path.isfile(file_path) and app in str(file_path):
                module = load_module_from_file(file_path)
                module_name = os.path.basename(file_path).replace(".py", "")

                for func_name, func in inspect.getmembers(module, inspect.isfunction):
                    process_function(app, module_name, func_name, func, swagger, module, rel_path_dotted, tag)
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            frappe.log_error(f"Error loading or processing file {file_path}: {str(e)}")

    # Define the path to the Swagger JSON file
    www_dir = os.path.join(frappe_bench_dir, "apps", "swagger", "swagger", "www")

    # Ensure the www directory exists
    if not os.path.exists(www_dir):
        os.makedirs(www_dir)

    # Save the generated Swagger JSON to a file
    file_path = os.path.join(www_dir, "swagger.json")
    try:
        # Save the generated Swagger JSON to a file
        print(f"[Swagger] Writing to: {file_path}")
        with open(file_path, "w") as swagger_file:
            json.dump(swagger, swagger_file, indent=4)
        print("[Swagger] Swagger JSON generated successfully.")
        
        # Return the swagger URL on success
        swagger_url = f"{site_url}/swagger"
        return {
            "success": True,
            "message": "Swagger JSON generated successfully",
            "swagger_url": swagger_url
        }
    except Exception as e:
        print(f"[Swagger] Failed to write swagger.json: {e}")
        frappe.log_error(f"Failed to write swagger.json: {e}")
        return {
            "success": False,
            "message": f"Failed to generate Swagger JSON: {str(e)}"
        }

## Generate Swagger json on reload
generate_swagger_json()