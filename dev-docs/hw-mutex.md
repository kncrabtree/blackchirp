# HardwareManager Mutex Refactoring Plan

## Current Problems
1. **Single mutex protecting multiple resources**: `d_accessMutex` protects both `d_hardwareMap` and connection state, causing deadlocks
2. **Deadlock chain**: `bcInitInstrument` → `bcTestConnection` → `handleConnectionResult` → tries to lock same mutex
3. **Read-heavy access patterns**: Most hardware map access is read-only (finding hardware), but uses exclusive locks

## Proposed Solution: Multi-Lock Architecture

### 1. Replace `d_accessMutex` with `QReadWriteLock d_hardwareMapLock`
- **Benefits**: Multiple threads can read hardware map simultaneously
- **Usage**: Read lock for finds/iterations, write lock for add/remove operations
- **Impact**: Better concurrency for hardware lookups during normal operation

### 2. Separate `QMutex d_connectionStateLock` for connection tracking
- **Purpose**: Protect `d_connectionState` independently from hardware map
- **Benefits**: Connection testing won't block hardware map access
- **Usage**: Only acquire when modifying connection counters/flags

### 3. Update `ConnectionTestState` to be fully thread-safe
- Keep existing `std::atomic` members for lock-free access where possible
- Use `d_connectionStateLock` only for compound operations

## Implementation Steps

### Phase 1: Infrastructure Changes
1. Replace `mutable QMutex d_accessMutex` with:
   - `mutable QReadWriteLock d_hardwareMapLock`  
   - `mutable QMutex d_connectionStateLock`
2. Update all template methods (`findHardware`, `findHardwareByType`) to use read locks
3. Update `resolveGpibController` to use read lock instead of full mutex

### Phase 2: Hardware Map Operations  
1. Update `addHardwareInternal`/`removeHardwareInternal`/`replaceHardwareInternal` to use write locks
2. Update `syncWithRuntimeConfig` difference detection to use read locks
3. Ensure proper lock ordering to prevent deadlocks

### Phase 3: Connection State Isolation
1. Update `handleConnectionResult` to use `d_connectionStateLock` only
2. Update `resetConnectionTestState`/`finalizeConnectionTesting` for new mutex
3. Remove connection state access from hardware map operations

### Phase 4: Remove Deadlock Sources
1. Remove `bcTestConnection` call from `bcInitInstrument` (let `testAll()` handle it)
2. Ensure `testAll()` releases hardware map lock before invoking connection tests
3. Fix `buildCommunication(parent())` calls separately

## Expected Benefits
- **Eliminates current deadlocks**: Connection testing independent from hardware map access
- **Better concurrency**: Multiple threads can read hardware map simultaneously  
- **Cleaner separation**: Each resource has appropriate protection level
- **Future-proof**: Easier to add more specialized hardware operations

## Implementation Status

### Current Progress
- **Analysis Complete**: Identified deadlock sources and mutex design flaws
- **Plan Approved**: Multi-lock architecture design reviewed and approved

### Next Steps
- Begin Phase 1 implementation with agent delegation
- Update task 3.3.7 to reference this document for tracking progress
