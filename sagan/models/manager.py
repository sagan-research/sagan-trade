import os
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger("sagan.manager")

class PerformanceProfile:
    ECO = "eco"
    BALANCED = "balanced"
    TURBO = "turbo"

PROFILES = {
    PerformanceProfile.ECO: {
        "ram_fraction": 0.10,
        "priority": "below_normal",
        "description": "Eco Mode: Low impact, keeps your PC smooth for other work.",
        "max_workers": 2,
    },
    PerformanceProfile.BALANCED: {
        "ram_fraction": 0.30,
        "priority": "normal",
        "description": "Balanced Mode: Fast training without freezing the PC.",
        "max_workers": max(1, os.cpu_count() // 2),
    },
    PerformanceProfile.TURBO: {
        "ram_fraction": 0.60,
        "priority": "above_normal",
        "description": "Turbo Mode: Maximum power for fastest results.",
        "max_workers": os.cpu_count(),
    }
}

class ResourceManager:
    """
    Handles OS-level optimizations and resource budgeting.
    """
    def __init__(self, profile_name: str = "balanced"):
        self.profile = PROFILES.get(profile_name, PROFILES["balanced"])
        self.total_ram = psutil.virtual_memory().total
        self.budget_ram = self.total_ram * self.profile["ram_fraction"]
        
    def apply_optimizations(self):
        """
        Applies process priority and other OS-level tuning.
        """
        p = psutil.Process(os.getpid())
        try:
            if os.name == 'nt': # Windows
                priority_map = {
                    "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    "normal": psutil.NORMAL_PRIORITY_CLASS,
                    "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                }
                p.nice(priority_map.get(self.profile["priority"], psutil.NORMAL_PRIORITY_CLASS))
            else: # Unix
                priority_map = {
                    "below_normal": 10,
                    "normal": 0,
                    "above_normal": -10,
                }
                os.nice(priority_map.get(self.profile["priority"], 0))
            logger.info(f"Applied {self.profile['priority']} priority.")
        except Exception as e:
            logger.warning(f"Could not sets process priority: {e}")

    def get_worker_count(self) -> int:
        return self.profile["max_workers"]

    def check_memory_safeguard(self) -> bool:
        """
        Returns False if memory usage exceeds the budget.
        """
        current_usage = psutil.Process(os.getpid()).memory_info().rss
        if current_usage > self.budget_ram:
            logger.warning("Memory budget exceeded!")
            return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "ram_used_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
            "ram_budget_mb": self.budget_ram / (1024 * 1024),
            "cpu_percent": psutil.cpu_percent(),
            "profile": self.profile["description"],
        }
