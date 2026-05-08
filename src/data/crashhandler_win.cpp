#include "crashhandler_p.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <signal.h>
#include <stdlib.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QString>

namespace {

std::atomic<int> g_inHandler{0};

// UTF-16 path of the most-recent dump prefix (without extension), used
// to derive both the .dmp and the .log sidecar filenames at fault time.
// Built once per reopen() call from non-handler context. Stored as a
// fixed buffer so the filter does not need to allocate.
constexpr std::size_t kPathBufWChars = 1024;
wchar_t g_dumpPathPrefix[kPathBufWChars]{};
wchar_t g_dumpPath[kPathBufWChars]{};
wchar_t g_logPath[kPathBufWChars]{};

void writeAll(HANDLE h, const char *buf, std::size_t len)
{
    DWORD written = 0;
    while(len > 0)
    {
        if(!::WriteFile(h, buf, static_cast<DWORD>(len), &written, nullptr) || written == 0)
            return;
        buf += written;
        len -= written;
    }
}

void writeStr(HANDLE h, const char *s)
{
    if(!s) return;
    writeAll(h, s, std::strlen(s));
}

const char *codeName(DWORD code)
{
    switch(code)
    {
        case EXCEPTION_ACCESS_VIOLATION:    return "EXCEPTION_ACCESS_VIOLATION";
        case EXCEPTION_STACK_OVERFLOW:      return "EXCEPTION_STACK_OVERFLOW";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:  return "EXCEPTION_INT_DIVIDE_BY_ZERO";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:  return "EXCEPTION_FLT_DIVIDE_BY_ZERO";
        case EXCEPTION_ILLEGAL_INSTRUCTION: return "EXCEPTION_ILLEGAL_INSTRUCTION";
        case EXCEPTION_PRIV_INSTRUCTION:    return "EXCEPTION_PRIV_INSTRUCTION";
        case EXCEPTION_IN_PAGE_ERROR:       return "EXCEPTION_IN_PAGE_ERROR";
        case EXCEPTION_BREAKPOINT:          return "EXCEPTION_BREAKPOINT";
        default:                            return "EXCEPTION_UNKNOWN";
    }
}

void writeMiniDump(EXCEPTION_POINTERS *ep)
{
    if(g_dumpPath[0] == L'\0')
        return;

    HANDLE file = ::CreateFileW(g_dumpPath,
                                GENERIC_WRITE,
                                0,
                                nullptr,
                                CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if(file == INVALID_HANDLE_VALUE)
        return;

    MINIDUMP_EXCEPTION_INFORMATION info{};
    info.ThreadId = ::GetCurrentThreadId();
    info.ExceptionPointers = ep;
    info.ClientPointers = FALSE;

    const auto type = static_cast<MINIDUMP_TYPE>(
        MiniDumpWithDataSegs |
        MiniDumpWithThreadInfo |
        MiniDumpWithProcessThreadData);

    ::MiniDumpWriteDump(::GetCurrentProcess(),
                        ::GetCurrentProcessId(),
                        file,
                        type,
                        ep ? &info : nullptr,
                        nullptr, nullptr);

    ::CloseHandle(file);
}

void writeSidecar(EXCEPTION_POINTERS *ep)
{
    if(g_logPath[0] == L'\0')
        return;

    HANDLE file = ::CreateFileW(g_logPath,
                                GENERIC_WRITE,
                                0,
                                nullptr,
                                CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if(file == INVALID_HANDLE_VALUE)
        return;

    writeStr(file, BC::CrashInternal::buildHeaderUtf8());

    SYSTEMTIME st;
    ::GetSystemTime(&st);
    char tsBuf[64];
    std::snprintf(tsBuf, sizeof(tsBuf),
                  "Crashed at %04u-%02u-%02uT%02u:%02u:%02uZ\n",
                  st.wYear, st.wMonth, st.wDay,
                  st.wHour, st.wMinute, st.wSecond);
    writeStr(file, tsBuf);

    if(ep && ep->ExceptionRecord)
    {
        char excBuf[256];
        const auto *r = ep->ExceptionRecord;
        std::snprintf(excBuf, sizeof(excBuf),
                      "Exception: %s (0x%08lX) at address 0x%p\n",
                      codeName(r->ExceptionCode),
                      static_cast<unsigned long>(r->ExceptionCode),
                      r->ExceptionAddress);
        writeStr(file, excBuf);
    }
    else
    {
        writeStr(file, "Exception: (no exception record)\n");
    }

    char metaBuf[128];
    std::snprintf(metaBuf, sizeof(metaBuf),
                  "PID: %lu\nActive experiment: %d\n",
                  static_cast<unsigned long>(::GetCurrentProcessId()),
                  BC::CrashInternal::activeExperimentNumber());
    writeStr(file, metaBuf);

    writeStr(file, "Minidump: see sibling .dmp file; resolve in WinDbg / VS\n"
                   "with the matching .pdb (identified by the build SHA in the\n"
                   "Blackchirp header line above).\n");

    ::FlushFileBuffers(file);
    ::CloseHandle(file);
}

LONG WINAPI topLevelFilter(EXCEPTION_POINTERS *ep)
{
    if(g_inHandler.fetch_add(1, std::memory_order_acq_rel) > 0)
        return EXCEPTION_CONTINUE_SEARCH;

    writeMiniDump(ep);
    writeSidecar(ep);

    return EXCEPTION_EXECUTE_HANDLER;
}

void __cdecl invalidParameter(const wchar_t * /*expr*/,
                              const wchar_t * /*func*/,
                              const wchar_t * /*file*/,
                              unsigned int    /*line*/,
                              uintptr_t       /*reserved*/)
{
    // Force the unhandled-exception path; the top-level filter will
    // do the real work.
    ::RaiseException(EXCEPTION_NONCONTINUABLE_EXCEPTION,
                     EXCEPTION_NONCONTINUABLE, 0, nullptr);
}

void __cdecl pureCallHandler()
{
    ::RaiseException(EXCEPTION_NONCONTINUABLE_EXCEPTION,
                     EXCEPTION_NONCONTINUABLE, 0, nullptr);
}

void abortHandler(int /*sig*/)
{
    ::RaiseException(EXCEPTION_NONCONTINUABLE_EXCEPTION,
                     EXCEPTION_NONCONTINUABLE, 0, nullptr);
}

}

namespace BC::CrashInternal {

void installPlatformHandlers()
{
    ::SetUnhandledExceptionFilter(topLevelFilter);
    ::_set_invalid_parameter_handler(invalidParameter);
    ::_set_purecall_handler(pureCallHandler);
    ::signal(SIGABRT, abortHandler);
}

void reopenPlatform()
{
    const char *dir = crashesDirUtf8();
    const char *base = currentBasenameUtf8();
    if(!dir || !*dir || !base || !*base)
        return;

    auto baseFull = QString::fromUtf8(dir) + QString::fromUtf8(base);
    auto baseW = baseFull.toStdWString();
    if(baseW.size() + 5 >= kPathBufWChars)
        return;

    std::wmemset(g_dumpPathPrefix, 0, kPathBufWChars);
    std::wmemcpy(g_dumpPathPrefix, baseW.data(), baseW.size());

    std::wmemset(g_dumpPath, 0, kPathBufWChars);
    std::wmemcpy(g_dumpPath, baseW.data(), baseW.size());
    std::wmemcpy(g_dumpPath + baseW.size(), L".dmp", 4);

    std::wmemset(g_logPath, 0, kPathBufWChars);
    std::wmemcpy(g_logPath, baseW.data(), baseW.size());
    std::wmemcpy(g_logPath + baseW.size(), L".log", 4);
}

void shutdownPlatform()
{
    g_dumpPath[0]       = L'\0';
    g_logPath[0]        = L'\0';
    g_dumpPathPrefix[0] = L'\0';
}

}
