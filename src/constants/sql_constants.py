from types import SimpleNamespace


TOKENS = ['<UNK>', '<BEG>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT']

KEYWORDS = {'SELECT': 'SELECT', 'FROM': 'FROM', 'WHERE': 'WHERE'}
KEYWORDS = SimpleNamespace(**KEYWORDS)

COND = {'EQUAL': 'EQL', 'GREATER_THAN': 'GT', 'LESS_THAN': 'LT'}
COND_OPERATORS = COND.values()
COND = SimpleNamespace(**COND)

AGG = {'NONE': 'None', 'MAX': 'MAX', 'MIN': 'MIN', 'COUNT': 'COUNT', 'SUM': 'SUM', 'AVERAGE': 'AVG'}
AGG_OPERATORS = AGG.values()
AGG = SimpleNamespace(**AGG)

