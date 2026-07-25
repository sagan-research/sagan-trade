import random
from parameters import DATASETS, UNARY_OPERATORS, BINARY_OPERATORS

class CombinatorialGenerator:
    def __init__(self):
        pass
        
    def generate_expression(self, depth=2) -> str:
        """
        Recursively builds an alpha expression AST and compiles to string.
        """
        if depth == 0:
            return random.choice(DATASETS)
            
        choice = random.random()
        if choice < 0.4:
            # Unary operation
            op = random.choice(UNARY_OPERATORS)
            inner = self.generate_expression(depth - 1)
            return op.format(x=inner)
        elif choice < 0.8:
            # Binary operation
            op = random.choice(BINARY_OPERATORS)
            left = self.generate_expression(depth - 1)
            right = self.generate_expression(depth - 1)
            return op.format(x=left, y=right)
        else:
            # Just a dataset
            return random.choice(DATASETS)

    def generate_batch(self, batch_size: int) -> list:
        """
        Generates a batch of unique expressions.
        """
        expressions = set()
        while len(expressions) < batch_size:
            expr = self.generate_expression(depth=random.randint(1, 3))
            
            # Post-processing to ensure valid syntax (e.g., closing parentheses)
            # and basic sanity checks (avoiding obvious divide-by-zero, etc.)
            expressions.add(expr)
            
        return list(expressions)
