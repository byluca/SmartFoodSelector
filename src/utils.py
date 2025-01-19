# src/utils.py

def print_banner(msg):
    """
    Stampa un messaggio incorniciato da caratteri '='.

    Args:
        msg (str): Il messaggio da stampare.

    Example:
        >>> print_banner("Hello")
        ==================================================
        Hello
        ==================================================
    """
    print("="*50)
    print(msg)
    print("="*50)
