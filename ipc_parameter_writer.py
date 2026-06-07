import ctypes
from ctypes import wintypes
import sys

FILE_MAP_ALL_ACCESS = 0xF001F
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = 0xFFFFFFFFFFFFFFFF
SHM_NAME = "Local\\SaganStrategySHM"

class TickerParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("ticker", ctypes.c_char * 8),                 # offset 0
        ("alpha_skew_factor", ctypes.c_double),        # offset 8
        ("max_trade_size", ctypes.c_uint32),           # offset 16
        ("inventory_limit", ctypes.c_int32),           # offset 20
        ("is_mean_reverting", ctypes.c_bool),          # offset 24
        ("padding_align", ctypes.c_char * 7),          # offset 25
        ("inventory_risk_gamma", ctypes.c_double),     # offset 32
        ("volatility_sigma", ctypes.c_double),         # offset 40
        ("fee_barrier_bps", ctypes.c_double),          # offset 48
        ("padding", ctypes.c_char * 8),                # offset 56
    ]

class SharedStrategyParams(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("tickers", TickerParams * 16),
        ("update_timestamp", ctypes.c_uint64),
        ("padding", ctypes.c_char * 56), # Align to exactly 1088 bytes (C++ alignas(64) padding)
    ]

class IPCParameterWriter:
    def __init__(self):
        kernel32 = ctypes.windll.kernel32
        
        # Configure argument and return types to prevent 64-bit truncation crashes
        kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, ctypes.c_wchar_p]
        kernel32.OpenFileMappingW.restype = wintypes.HANDLE
        
        kernel32.CreateFileMappingW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_wchar_p]
        kernel32.CreateFileMappingW.restype = wintypes.HANDLE
        
        kernel32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
        kernel32.MapViewOfFile.restype = ctypes.c_void_p
        
        kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        kernel32.UnmapViewOfFile.restype = wintypes.BOOL
        
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        # 1. Attempt to open existing mapping
        self.hMap = kernel32.OpenFileMappingW(
            FILE_MAP_ALL_ACCESS,
            False,
            SHM_NAME
        )
        
        # 2. If mapping doesn't exist, create a new one
        if not self.hMap or self.hMap == wintypes.HANDLE(0).value:
            self.hMap = kernel32.CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                None,
                PAGE_READWRITE,
                0,
                ctypes.sizeof(SharedStrategyParams),
                SHM_NAME
            )
            
        if not self.hMap or self.hMap == wintypes.HANDLE(0).value:
            raise RuntimeError(f"Failed to open or create Shared Memory Mapping '{SHM_NAME}'. Error code: {kernel32.GetLastError()}")
            
        # 3. Map View of File
        self.pBuf = kernel32.MapViewOfFile(
            self.hMap,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            ctypes.sizeof(SharedStrategyParams)
        )
        
        if not self.pBuf or self.pBuf == 0:
            kernel32.CloseHandle(self.hMap)
            self.hMap = None
            raise RuntimeError(f"Failed to map view of Shared Memory. Error code: {kernel32.GetLastError()}")
            
        # 4. Map the Ctypes structure to the memory address pointer
        self.shared_params = SharedStrategyParams.from_address(self.pBuf)

    def update_ticker(self, ticker: str, skew: float, size: int, mean_revert: bool,
                      inventory_limit: int = 150, inventory_risk_gamma: float = 0.05,
                      volatility_sigma: float = 0.01, fee_barrier_bps: float = 3.52):
        if not self.pBuf or self.pBuf == 0:
            raise RuntimeError("Cannot write parameters: Memory mapping is inactive.")
            
        target_slot = -1
        empty_slot = -1
        ticker_bytes = ticker.encode('utf-8')[:7]
        
        for i in range(16):
            slot_ticker = self.shared_params.tickers[i].ticker
            try:
                slot_ticker_str = slot_ticker.decode('utf-8').rstrip('\x00')
            except:
                slot_ticker_str = ""
            
            if slot_ticker_str == ticker:
                target_slot = i
                break
            if empty_slot == -1 and (len(slot_ticker_str) == 0 or slot_ticker[0] == 0):
                empty_slot = i
                
        slot = target_slot if target_slot != -1 else (empty_slot if empty_slot != -1 else 0)
        
        # Write packed data structures to memory address offset
        self.shared_params.tickers[slot].ticker = ticker_bytes + b'\x00' * (8 - len(ticker_bytes))
        self.shared_params.tickers[slot].alpha_skew_factor = skew
        self.shared_params.tickers[slot].max_trade_size = size
        self.shared_params.tickers[slot].inventory_limit = inventory_limit
        self.shared_params.tickers[slot].is_mean_reverting = mean_revert
        self.shared_params.tickers[slot].inventory_risk_gamma = inventory_risk_gamma
        self.shared_params.tickers[slot].volatility_sigma = volatility_sigma
        self.shared_params.tickers[slot].fee_barrier_bps = fee_barrier_bps
        self.shared_params.update_timestamp += 1
        
        print(f"[IPC SHM Writer] Updated slot {slot} for {ticker}: skew={skew:.4f}, size={size}, mean_revert={mean_revert}, inv_limit={inventory_limit}, gamma={inventory_risk_gamma:.4f}, sigma={volatility_sigma:.4f}, fee_barrier={fee_barrier_bps:.2f}")

    def __del__(self):
        kernel32 = ctypes.windll.kernel32
        if hasattr(self, 'pBuf') and self.pBuf and self.pBuf != 0:
            kernel32.UnmapViewOfFile(self.pBuf)
            self.pBuf = None
        if hasattr(self, 'hMap') and self.hMap:
            kernel32.CloseHandle(self.hMap)
            self.hMap = None

if __name__ == "__main__":
    writer = IPCParameterWriter()
    writer.update_ticker("RELIANCE", 0.12, 100, False)
    writer.update_ticker("INFY", 0.05, 50, True)
    print("Test passed: Shared memory parameters updated successfully!")
