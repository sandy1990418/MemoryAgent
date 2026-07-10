"""Mock tools shared by the runnable agent demos."""

from __future__ import annotations

import ast
import operator

from langchain.tools import tool


@tool
def weather(city: str) -> str:
    """Return a tiny mock weather report for a city."""
    return f"{city}: sunny, 26 C, light wind. This is mock data for the demo."


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    allowed_binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary_ops:
            return allowed_binary_ops[type(node.op)](
                eval_node(node.left),
                eval_node(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
            return allowed_unary_ops[type(node.op)](eval_node(node.operand))
        raise ValueError("Only basic arithmetic is supported.")

    return str(eval_node(ast.parse(expression, mode="eval")))


DEMO_TOOLS = [weather, calculator]

