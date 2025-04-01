#!/usr/bin/env python3
"""
Pattern Matching Example

This file contains code snippets with similar structures but different:
- Variable names
- Formatting
- Literal values
- Comments
- Minor logic variations

These snippets are designed to demonstrate how AST-based pattern matching
can find structural similarities that text-based matching would miss.
"""

from multi_agent_crew_automation_for_code_quality_improvement.algorithm_tools import (
    FuzzyPatternMatchingTool,
)


# Example 1: User Authentication - Verbose version with descriptive names
def validate_user_credentials(username, password, max_attempts=3):
    """Authenticate a user with username and password."""
    attempts = 0
    is_authenticated = False
    
    while attempts < max_attempts and not is_authenticated:
        if len(username) < 3 or len(password) < 8:
            print("Username or password too short")
            return False
            
        # Check if the credentials match our database
        if username == "admin" and password == "secure_password123":
            is_authenticated = True
            print("Authentication successful!")
        else:
            attempts += 1
            print(f"Authentication failed. {max_attempts - attempts} attempts remaining.")
    
    if not is_authenticated:
        print("Maximum authentication attempts exceeded. Account locked.")
        
    return is_authenticated


# Example 2: Product Inventory - Completely different variable names and formatting
def check_stock(product_id,quantity_requested,  reorder_threshold = 5):
    """Check if a product is available in sufficient quantity."""
    # Initialize variables
    current_quantity=0
    available=False
    
    # Check inventory until product found or we've checked everything
    while current_quantity==0 and not available:
        if product_id=="" or quantity_requested<=0: 
            print("Invalid product or quantity")
            return False
        # Look up in inventory database
        if product_id=="ABC123" and quantity_requested<=10:
            available=True
            print("Product available!")
        else:
            current_quantity+=1 
            print("Checking alternative inventory locations: "+str(current_quantity))
    
    if not available:
        print("Product not available in requested quantity.")
    return available


# Example 3: Email Validation - Functionally similar but with different logic and formatting
def is_email_valid(email_address, allow_domains=None, max_checks=3):
    """Validate an email address format and domain."""
    validation_count = 0
    email_valid = False
    
    # Perform validation checks
    while validation_count < max_checks and not email_valid:
        # Basic format validation
        if '@' not in email_address or '.' not in email_address:
            print("Email must contain @ and .")
            return False
        
        # Extract domain and check against allowed list
        domain = email_address.split('@')[1]
        if allow_domains and domain in allow_domains:
            email_valid = True
            print("Email domain is approved")
        else:
            validation_count += 1
            print(f"Performing additional validation: {validation_count}/{max_checks}")
    
    if not email_valid:
        print("Email validation failed after multiple checks")
        
    return email_valid


# Example 4: Completely different domain but same structure
def process_payment(amount, payment_method, retry_count=3):
    """Process a payment transaction."""
    attempts = 0
    payment_successful = False
    
    # Try to process payment
    while attempts < retry_count and not payment_successful:
        # Validate input parameters
        if amount <= 0 or not payment_method:
            print("Invalid payment amount or method")
            return False
        
        # Try to process the transaction
        if payment_method == "credit_card" and amount < 1000:
            payment_successful = True
            print("Payment processed successfully")
        else:
            attempts += 1
            print(f"Payment processing failed. Retrying ({attempts}/{retry_count})")
    
    if not payment_successful:
        print("Payment processing failed after multiple attempts")
        
    return payment_successful


def run_pattern_matching_demo():
    """Run the pattern matching demonstration."""
    # Extract the code of each function as strings
    import inspect
    
    example1 = inspect.getsource(validate_user_credentials)
    example2 = inspect.getsource(check_stock)
    example3 = inspect.getsource(is_email_valid)
    example4 = inspect.getsource(process_payment)
    
    # Initialize the pattern matching tool
    pattern_matcher = FuzzyPatternMatchingTool()
    
    print("\n== TEXT-BASED COMPARISON ==")
    result_text = pattern_matcher._run(
        code_snippets=[example1, example2, example3, example4],
        similarity_threshold=0.3  # Lower threshold to see some results
    )
    print(result_text)
    
    print("\n== TOKEN-BASED COMPARISON ==")
    result_token = pattern_matcher._run(
        code_snippets=[example1, example2, example3, example4],
        token_based=True,
        similarity_threshold=0.4  # Lower threshold to see some results
    )
    print(result_token)
    
    print("\n== AST-BASED COMPARISON ==")
    result_ast = pattern_matcher._run(
        code_snippets=[example1, example2, example3, example4],
        ast_based=True,
        similarity_threshold=0.7  # Higher threshold works with AST comparison
    )
    print(result_ast)


if __name__ == "__main__":
    run_pattern_matching_demo()
