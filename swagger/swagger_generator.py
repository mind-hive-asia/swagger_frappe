import ast
import importlib.util
import inspect
import json
import os
import glob
from typing import Optional

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

def process_function(app_name, module_name, func_name, func, swagger, module, rel_path_dotted=None, tag = None):
    """Process each function to update the Swagger paths.
    
    Args:
        app_name (str): The name of the app.
        module_name (str): The name of the module.
        func_name (str): The name of the function being processed.
        func (function): The function object.
        swagger (dict): The Swagger specification to be updated.
        module (module): The module where the function is defined.
        rel_path_dotted (str, optional): The relative dotted path from the endpoint folder (without .py).
    """
    try:
        source_code = inspect.getsource(func)
        tree = ast.parse(source_code)

        # Skip functions that do not contain validate_http_method calls
        if not any(
            ("validate_http_method" in ast.dump(node) or "rest_handler" in ast.dump(node)) and isinstance(node, ast.Call)
            for node in ast.walk(tree)
        ):
            print(f"Skipping {func_name}: 'validate_http_method' not found")
            return

        # Find the Pydantic model used in the validate_request decorator
        pydantic_model_name = find_pydantic_model_in_decorator(tree)

        # Construct the API path for the function
        if rel_path_dotted:
            path = f"/api/method/{app_name}.{rel_path_dotted}.{func_name}".lower()
        else:
            path = f"/api/method/{app_name}.api.{module_name}.{func_name}".lower()

        # Define the mapping of HTTP methods to check for in the source code
        http_methods = {
            "GET": "GET",
            "POST": "POST",
            "PUT": "PUT",
            "DELETE": "DELETE",
            "PATCH": "PATCH",
            "OPTIONS": "OPTIONS",
            "HEAD": "HEAD",
        }

        # Default HTTP method is POST
        http_method = "POST"
        for method in http_methods:
            if method in source_code:
                http_method = method
                break

        # Define the request body for methods that modify data
        request_body = {}
        if pydantic_model_name and http_method in ["POST", "PUT", "PATCH"]:
            pydantic_schema = get_pydantic_model_schema(pydantic_model_name, module)
            if pydantic_schema:
                request_body = {
                    "description": "Request body",
                    "required": True,
                    "content": {"application/json": {"schema": pydantic_schema}},
                }

        # Define query parameters for methods that retrieve data
        params = []
        if http_method in ["GET", "DELETE", "OPTIONS", "HEAD"]:
            signature = inspect.signature(func)
            for param_name, param in signature.parameters.items():
                if (
                    param.default is inspect.Parameter.empty
                    and not "kwargs" in param_name
                ):
                    param_type = "string"
                    params.append(
                        {
                            "name": param_name,
                            "in": "query",
                            "required": True,
                            "schema": {"type": param_type},
                        }
                    )

        # Define the response schema
        responses = {
            "200": {
                "description": "Successful response",
                "content": {"application/json": {"schema": {"type": "object"}}},
            }
        }

        # Assign tags for the Swagger documentation
        tags = [module_name if not tag else tag]

        # Get the function docstring for description
        docstring = inspect.getdoc(func) or ""

        # Detect allow_guest=True in @frappe.whitelist decorator
        allow_guest = False
        for decorator in tree.body[0].decorator_list:
            if isinstance(decorator, ast.Call):
                if (
                    (isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "whitelist") or
                    (isinstance(decorator.func, ast.Name) and decorator.func.id == "frappe.whitelist")
                ):
                    for kw in decorator.keywords:
                        if kw.arg == "allow_guest" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            allow_guest = True

        # Initialize the path if not already present
        if path not in swagger["paths"]:
            swagger["paths"][path] = {}

        # Build the operation object
        operation = {
            "summary": func_name.title().replace("_", " "),
            "description": docstring,
            "tags": tags,
            "parameters": params,
            "requestBody": request_body if request_body else None,
            "responses": responses,
        }
        if not allow_guest:
            operation["security"] = [{"basicAuth": []}]

        swagger["paths"][path][http_method.lower()] = operation
    except Exception as e:
        # Log any errors that occur during processing
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
            (f"apps/{app_name}/{app_name}/{app_name}/core/endpoints/v1", "Core - "),
            (f"apps/{app_name}/{app_name}/{app_name}/endpoints/v1/", "Application - ")
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
                    tag = f"{tag_prefix}{os.path.relpath(os.path.dirname(file_path), abs_folder).replace('_', ' ').title()}"
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