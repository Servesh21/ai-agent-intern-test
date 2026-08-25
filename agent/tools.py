"""Tool definitions and dispatcher for Gemini function calling."""

from typing import Any
from google.genai import types
from agent.orders import lookup_order


def get_agent_tools() -> list[Any]:
    """Get the list of tools for Gemini generation config.
    
    Using types.FunctionDeclaration for deterministic schema specification.
    """
    lookup_order_func = types.FunctionDeclaration(
        name="lookup_order",
        description=(
            "Look up an Aster & Row order by its order ID (e.g. 'ORD-1007'). "
            "Returns customer-safe details including status, items, tracking, and delivery estimate. "
            "Internal notes, email, shipping address, and risk scores are omitted for privacy."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "order_id": types.Schema(
                    type=types.Type.STRING,
                    description="The order ID to look up, such as 'ORD-1001' or 'ORD-1007'.",
                ),
            },
            required=["order_id"],
        ),
    )

    return [types.Tool(function_declarations=[lookup_order_func])]


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with provided arguments and return a customer-safe result."""
    if name == "lookup_order":
        order_id = args.get("order_id", "")
        return lookup_order(order_id)
    return {
        "success": False,
        "error": f"Unknown tool: '{name}'",
        "error_type": "unknown_tool",
    }
