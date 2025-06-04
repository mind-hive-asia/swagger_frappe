import ast
import importlib.util
import inspect
import json
import os
import glob
from typing import get_origin, get_args, List, Optional, Dict, Any, Union


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

def process_function(app_name: str,
                     module_name: str,
                     func_name: str,
                     func: Any,
                     swagger: Dict[str, Any],
                     module: Any,
                     rel_path_dotted: str = None,
                     tag: str = None):
    """
    Process each function to update the Swagger paths, using Python type hints and
    docstrings for parameter/response schemas and descriptions.

    Args:
        app_name (str): The name of the app.
        module_name (str): The name of the module.
        func_name (str): The name of the function being processed.
        func (callable): The function object.
        swagger (dict): The Swagger (OpenAPI) spec to be updated.
        module (module): The module where the function is defined.
        rel_path_dotted (str, optional): The relative dotted path from the endpoint folder (without .py).
        tag (str, optional): Optional tag to group endpoints in Swagger.
    """
    try:
        source_code = inspect.getsource(func)
        tree = ast.parse(source_code)

        # Skip functions that do not contain validate_http_method or rest_handler calls
        if not any(
            (
                ("validate_http_method" in ast.dump(node) or "rest_handler" in ast.dump(node))
                and isinstance(node, ast.Call)
            )
            for node in ast.walk(tree)
        ):
            print(f"Skipping {func_name}: 'validate_http_method' not found")
            return

        # Construct the API path
        if rel_path_dotted:
            path = f"/api/method/{app_name}.{rel_path_dotted}.{func_name}".lower()
        else:
            path = f"/api/method/{app_name}.api.{module_name}.{func_name}".lower()

        # Determine HTTP method (default to POST)
        http_method = "POST"
        for candidate in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
            if candidate in source_code:
                http_method = candidate
                break

        # Prepare containers for parameters and requestBody
        parameters: List[Dict[str, Any]] = []
        request_body: Dict[str, Any] = {}

        # Inspect signature and type hints
        signature = inspect.signature(func)
        for param_name, param in signature.parameters.items():
            # Skip 'self' or 'cls' for methods
            if param_name in ("self", "cls"):
                continue

            annotation = param.annotation
            default = param.default

            # If annotation is a Pydantic model, treat as requestBody (for POST/PUT/PATCH)
            if (
                inspect.isclass(annotation)
                and issubclass(annotation, BaseModel)
                and http_method in ("POST", "PUT", "PATCH")
            ):
                # Only one BaseModel in signature is allowed for body; take the first
                model_schema = annotation.schema()
                request_body = {
                    "description": annotation.__doc__.strip() if annotation.__doc__ else "Request body",
                    "required": True if default is inspect._empty else False,
                    "content": {
                        "application/json": {
                            "schema": model_schema
                        }
                    },
                }
                # No need to add this parameter to 'parameters'; skip to next
                continue

            # Otherwise, treat as a query/path parameter for GET, DELETE, etc.
            if http_method in ("GET", "DELETE", "OPTIONS", "HEAD") or (
                http_method in ("POST", "PUT", "PATCH") and not (
                    inspect.isclass(annotation) and issubclass(annotation, BaseModel)
                )
            ):
                schema_type: Dict[str, Any] = {"type": "string"}
                description = ""

                # Map Python basic types to OpenAPI types
                if annotation != inspect._empty:
                    origin = get_origin(annotation)
                    args = get_args(annotation)

                    if origin is Union and type(None) in args:
                        # Optional[...] case
                        non_none_args = [a for a in args if a is not type(None)]
                        if len(non_none_args) == 1:
                            annotation = non_none_args[0]
                            origin = get_origin(annotation)
                            args = get_args(annotation)

                    if annotation in (str,):
                        schema_type = {"type": "string"}
                    elif annotation in (int,):
                        schema_type = {"type": "integer", "format": "int32"}
                    elif annotation in (float,):
                        schema_type = {"type": "number", "format": "float"}
                    elif annotation in (bool,):
                        schema_type = {"type": "boolean"}
                    elif origin in (list, List):
                        # e.g., List[int], List[str]
                        item_type = args[0] if args else str
                        item_schema: Dict[str, Any] = {"type": "string"}
                        if item_type is int:
                            item_schema = {"type": "integer", "format": "int32"}
                        elif item_type is float:
                            item_schema = {"type": "number", "format": "float"}
                        elif item_type is bool:
                            item_schema = {"type": "boolean"}
                        schema_type = {"type": "array", "items": item_schema}
                    else:
                        # Fallback: treat as string
                        schema_type = {"type": "string"}

                # Mark as required if no default is provided
                required = default is inspect._empty

                parameters.append({
                    "name": param_name,
                    "in": "query",
                    "required": required,
                    "schema": schema_type,
                    "description": description,
                })

        # Determine response schema from return annotation (if any)
        responses: Dict[str, Any] = {
            "200": {
                "description": "Successful response",
                "content": {"application/json": {"schema": {"type": "object"}}},
            }
        }
        return_annotation = signature.return_annotation
        if return_annotation != inspect._empty:
            if inspect.isclass(return_annotation) and issubclass(return_annotation, BaseModel):
                responses["200"]["content"]["application/json"]["schema"] = return_annotation.schema()
            else:
                # Map a few basic return types; otherwise leave as generic object
                if return_annotation is str:
                    responses["200"]["content"]["application/json"]["schema"] = {"type": "string"}
                elif return_annotation is int:
                    responses["200"]["content"]["application/json"]["schema"] = {"type": "integer", "format": "int32"}
                elif return_annotation is float:
                    responses["200"]["content"]["application/json"]["schema"] = {"type": "number", "format": "float"}
                elif return_annotation is bool:
                    responses["200"]["content"]["application/json"]["schema"] = {"type": "boolean"}
                # For List[...] return types, more elaborate parsing can be added here

        # Assign tags for grouping
        tags_list = [module_name if not tag else tag]

        # Get the function docstring for endpoint description
        description = inspect.getdoc(func) or ""

        # Detect allow_guest=True in @frappe.whitelist decorator
        allow_guest = False
        for decorator in tree.body[0].decorator_list:
            if isinstance(decorator, ast.Call):
                func_node = decorator.func
                if (
                    (isinstance(func_node, ast.Attribute) and func_node.attr == "whitelist")
                    or (isinstance(func_node, ast.Name) and func_node.id == "frappe.whitelist")
                ):
                    for kw in decorator.keywords:
                        if kw.arg == "allow_guest" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            allow_guest = True

        # Initialize path entry if absent
        if path not in swagger.setdefault("paths", {}):
            swagger["paths"][path] = {}

        # Build operation object
        operation: Dict[str, Any] = {
            "summary": func_name.replace("_", " ").title(),
            "description": description,
            "tags": tags_list,
            "parameters": parameters if parameters else [],
            "responses": responses,
        }
        if request_body:
            operation["requestBody"] = request_body
        if not allow_guest:
            operation["security"] = [{"basicAuth": []}]

        swagger["paths"][path][http_method.lower()] = operation

    except Exception as e:
        frappe.log_error(
            f"Error processing function {func_name} in module {module_name}: {str(e)}"
        )


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
    
    # Initialize the Swagger specification
    swagger = {
        "openapi": "3.0.0",
        "info": {
            "title": f"{swagger_settings.app_name} API",
            "version": "1.0.0",
        },
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
            (f"apps/{app_name}/{app_name}/{app_name}/endpoints/", "Application - "),
            (f"apps/{app_name}/{app_name}/{app_name}/core/endpoints/", "Core - "),
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
    except Exception as e:
        print(f"[Swagger] Failed to write swagger.json: {e}")
        frappe.log_error(f"Failed to write swagger.json: {e}")

## Generate Swagger json on reload
generate_swagger_json()