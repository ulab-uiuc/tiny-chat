import argparse

from flask import Flask, Response, jsonify, request

from tiny_chat.utils import TemplateManager

app = Flask(__name__)
prompt_manager = TemplateManager()


@app.route("/api/prompts", methods=["GET"])  # type: ignore[misc]
def list_templates() -> Response:
    """List all available templates"""
    try:
        templates = prompt_manager.list_templates()
        return jsonify({"success": True, "data": templates, "count": len(templates)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/names", methods=["GET"])  # type: ignore[misc]
def list_template_names() -> Response:
    """List all template names"""
    try:
        names = prompt_manager.list_template_names()
        return jsonify({"success": True, "data": names, "count": len(names)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/<template_name>", methods=["GET"])  # type: ignore[misc]
def get_template(template_name: str) -> Response:
    """Get a specific template"""
    try:
        content = prompt_manager.get_template(template_name)
        info = prompt_manager.get_template_info(template_name)
        return jsonify(
            {
                "success": True,
                "data": {"name": template_name, "content": content, "info": info},
            }
        )
    except KeyError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/<template_name>", methods=["PUT"])  # type: ignore[misc]
def update_template(template_name: str) -> Response:
    """Update an existing template"""
    try:
        data = request.get_json()
        if not data or "content" not in data:
            return jsonify({"success": False, "error": "Content is required"}), 400

        content = data["content"]
        save = data.get("save", True)

        prompt_manager.set_template(template_name, content, save)

        return jsonify(
            {
                "success": True,
                "message": f"Template {template_name} updated successfully",
                "data": {"name": template_name, "content": content},
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts", methods=["POST"])  # type: ignore[misc]
def add_template() -> Response:
    """Add a new template"""
    try:
        data = request.get_json()
        if not data or "name" not in data or "content" not in data:
            return (
                jsonify({"success": False, "error": "Name and content are required"}),
                400,
            )

        template_name = data["name"]
        content = data["content"]
        save = data.get("save", True)

        prompt_manager.add_template(template_name, content, save)

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Template {template_name} added successfully",
                    "data": {"name": template_name, "content": content},
                }
            ),
            201,
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 409
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/<template_name>", methods=["DELETE"])  # type: ignore[misc]
def remove_template(template_name: str) -> Response:
    """Remove a custom template"""
    try:
        data = request.get_json() or {}
        save = data.get("save", True)

        prompt_manager.remove_template(template_name, save)

        return jsonify(
            {
                "success": True,
                "message": f"Template {template_name} removed successfully",
            }
        )
    except (KeyError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/<template_name>/format", methods=["POST"])  # type: ignore[misc]
def format_template(template_name: str) -> Response:
    """Format a template with parameters"""
    try:
        data = request.get_json() or {}
        parameters = data.get("parameters", {})

        formatted = prompt_manager.format_template(template_name, **parameters)

        return jsonify(
            {
                "success": True,
                "data": {
                    "template_name": template_name,
                    "parameters": parameters,
                    "formatted_content": formatted,
                },
            }
        )
    except KeyError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/reset", methods=["POST"])  # type: ignore[misc]
def reset_templates() -> Response:
    """Reset all templates to defaults"""
    try:
        data = request.get_json() or {}
        save = data.get("save", True)

        prompt_manager.reset_to_defaults(save)

        return jsonify(
            {"success": True, "message": "All templates reset to default values"}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/export", methods=["GET"])  # type: ignore[misc]
def export_templates() -> Response:
    """Export all templates as JSON"""
    try:
        templates = prompt_manager.list_templates()
        return jsonify({"success": True, "data": templates})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/prompts/import", methods=["POST"])  # type: ignore[misc]
def import_templates() -> Response:
    """Import templates from JSON"""
    try:
        data = request.get_json()
        if not data or "templates" not in data:
            return (
                jsonify({"success": False, "error": "Templates data is required"}),
                400,
            )

        templates = data["templates"]
        overwrite = data.get("overwrite", False)
        save = data.get("save", True)

        imported_count = 0
        skipped_count = 0

        for name, content in templates.items():
            try:
                if overwrite:
                    prompt_manager.set_template(name, content, save=False)
                else:
                    prompt_manager.add_template(name, content, save=False)
                imported_count += 1
            except ValueError:
                skipped_count += 1

        if save:
            prompt_manager._save_custom_templates()

        return jsonify(
            {
                "success": True,
                "message": f"Imported {imported_count} templates, skipped {skipped_count}",
                "data": {"imported": imported_count, "skipped": skipped_count},
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])  # type: ignore[misc]
def health_check() -> Response:
    """Health check endpoint"""
    return jsonify(
        {
            "success": True,
            "status": "healthy",
            "template_count": len(prompt_manager.list_templates()),
        }
    )


@app.route("/", methods=["GET"])  # type: ignore[misc]
def index() -> Response:
    """API documentation"""
    return jsonify(
        {
            "name": "Tiny Chat Prompt API",
            "version": "1.0.0",
            "endpoints": {
                "GET /api/prompts": "List all templates",
                "GET /api/prompts/names": "List template names",
                "GET /api/prompts/<name>": "Get specific template",
                "PUT /api/prompts/<name>": "Update template",
                "POST /api/prompts": "Add new template",
                "DELETE /api/prompts/<name>": "Remove template",
                "POST /api/prompts/<name>/format": "Format template",
                "POST /api/prompts/reset": "Reset to defaults",
                "GET /api/prompts/export": "Export templates",
                "POST /api/prompts/import": "Import templates",
                "GET /api/health": "Health check",
            },
        }
    )


def main() -> None:
    """Main function to run the API server"""
    parser = argparse.ArgumentParser(description="Tiny Chat Prompt API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting Tiny Chat Prompt API Server on {args.host}:{args.port}")
    print("Available endpoints:")
    print("  GET  /api/prompts - List all templates")
    print("  GET  /api/prompts/names - List template names")
    print("  GET  /api/prompts/<name> - Get specific template")
    print("  PUT  /api/prompts/<name> - Update template")
    print("  POST /api/prompts - Add new template")
    print("  DELETE /api/prompts/<name> - Remove template")
    print("  POST /api/prompts/<name>/format - Format template")
    print("  POST /api/prompts/reset - Reset to defaults")
    print("  GET  /api/prompts/export - Export templates")
    print("  POST /api/prompts/import - Import templates")
    print("  GET  /api/health - Health check")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
