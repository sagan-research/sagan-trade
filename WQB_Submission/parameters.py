BATCH_SIZE = 100
MAX_CONCURRENT_REQUESTS = 10
RETRY_DELAY = 5
MAX_RETRIES = 3

# IS Pass Criteria
MIN_SHARPE = 1.25
MIN_TURNOVER = 0.01  # 1%
MAX_TURNOVER = 0.70  # 70%
MIN_FITNESS = 1.0

# Simulation Settings
REGION = "USA"
DELAY = 1
UNIVERSE = "TOP3000"
TRUNCATE = 0.08
NEUTRALIZATION = "SUBINDUSTRY"
DECAY = 0

# Alpha Generation Building Blocks
DATASETS = [
    "close", "open", "high", "low", "volume", "vwap", "returns"
]

UNARY_OPERATORS = [
    "ts_rank({x}, 10)",
    "ts_mean({x}, 10)",
    "ts_std_dev({x}, 10)",
    "decay_linear({x}, 10)",
    "rank({x})",
    "sign({x})",
    "scale({x})"
]

BINARY_OPERATORS = [
    "{x} + {y}",
    "{x} - {y}",
    "{x} * {y}",
    "{x} / ({y} + 0.0001)",
    "ts_corr({x}, {y}, 10)",
    "ts_covariance({x}, {y}, 10)"
]
