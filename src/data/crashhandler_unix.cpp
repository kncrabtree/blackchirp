#include "crashhandler_p.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <exception>

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// dladdr lives in <dlfcn.h>; available on Linux (glibc) and macOS.
#include <dlfcn.h>

// backtrace(3) is available on Linux (glibc) and macOS; included
// unconditionally so the fallback path always compiles, regardless of
// whether <stacktrace> is also available.
#include <execinfo.h>

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QString>
#include <QtGlobal>

#if __has_include(<stacktrace>) && defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
  #include <stacktrace>
  #define BC_HAVE_STD_STACKTRACE 1
#else
  #define BC_HAVE_STD_STACKTRACE 0
#endif

namespace {

constexpr std::size_t kAltStackSize = 64 * 1024;
alignas(16) char g_altStack[kAltStackSize];

std::atomic<int> g_logFd{-1};
QByteArray g_logPath;

// Recursion guard: a fault inside the handler must not loop.
std::atomic<int> g_inHandler{0};

void writeAll(int fd, const char *buf, std::size_t len)
{
    while(len > 0)
    {
        ssize_t n = ::write(fd, buf, len);
        if(n < 0)
        {
            if(errno == EINTR) continue;
            return;
        }
        buf += n;
        len -= static_cast<std::size_t>(n);
    }
}

void writeStr(int fd, const char *s)
{
    if(!s) return;
    writeAll(fd, s, std::strlen(s));
}

const char *signalName(int sig)
{
    switch(sig)
    {
        case SIGSEGV: return "SIGSEGV";
        case SIGABRT: return "SIGABRT";
        case SIGFPE:  return "SIGFPE";
        case SIGILL:  return "SIGILL";
        case SIGBUS:  return "SIGBUS";
        default:      return "SIG?";
    }
}

void emitFrame(int fd, void *pc)
{
    char buf[256];
    Dl_info info{};
    if(::dladdr(pc, &info) && info.dli_fbase)
    {
        const auto *name = (info.dli_fname && info.dli_fname[0]) ? info.dli_fname : "?";
        auto base = reinterpret_cast<std::uintptr_t>(info.dli_fbase);
        auto pcv  = reinterpret_cast<std::uintptr_t>(pc);
        auto off  = pcv - base;
        std::snprintf(buf, sizeof(buf), "  %s(+0x%lx) [0x%lx]\n",
                      name, static_cast<unsigned long>(off),
                      static_cast<unsigned long>(pcv));
    }
    else
    {
        std::snprintf(buf, sizeof(buf), "  ?(+0x0) [0x%lx]\n",
                      static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(pc)));
    }
    writeStr(fd, buf);
}

void emitStackTrace(int fd)
{
#if BC_HAVE_STD_STACKTRACE
    auto trace = std::stacktrace::current();
    for(const auto &entry : trace)
    {
        auto raw = entry.native_handle();
        void *pc = reinterpret_cast<void*>(static_cast<std::uintptr_t>(raw));
        emitFrame(fd, pc);
    }
#else
    void *frames[128];
    int n = ::backtrace(frames, 128);
    for(int i = 1; i < n; ++i)
        emitFrame(fd, frames[i]);
#endif
}

void writeCrashLog(int sig, siginfo_t *info)
{
    int fd = g_logFd.load(std::memory_order_acquire);
    if(fd < 0)
        return;

    writeStr(fd, BC::CrashInternal::buildHeaderUtf8());

    // ISO 8601 timestamp via gmtime_r — async-signal-safe in practice on
    // glibc and on Apple libc.
    std::time_t now = std::time(nullptr);
    std::tm tm{};
    ::gmtime_r(&now, &tm);
    char tsBuf[64];
    std::snprintf(tsBuf, sizeof(tsBuf),
                  "Crashed at %04d-%02d-%02dT%02d:%02d:%02dZ\n",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    writeStr(fd, tsBuf);

    char sigBuf[128];
    void *faultAddr = info ? info->si_addr : nullptr;
    std::snprintf(sigBuf, sizeof(sigBuf),
                  "Signal: %s (%d) at address 0x%lx\n",
                  signalName(sig), sig,
                  static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(faultAddr)));
    writeStr(fd, sigBuf);

    char metaBuf[128];
    std::snprintf(metaBuf, sizeof(metaBuf),
                  "PID: %ld\nActive experiment: %d\n\nStack trace:\n",
                  static_cast<long>(::getpid()),
                  BC::CrashInternal::activeExperimentNumber());
    writeStr(fd, metaBuf);

    emitStackTrace(fd);

    ::fsync(fd);
}

void crashHandler(int sig, siginfo_t *info, void * /*ucontext*/)
{
    if(g_inHandler.fetch_add(1, std::memory_order_acq_rel) > 0)
    {
        // Already running — bail to the default disposition.
        ::signal(sig, SIG_DFL);
        ::raise(sig);
        return;
    }

    writeCrashLog(sig, info);

    // Restore the default disposition and re-raise so the kernel can
    // produce a core dump for users with `ulimit -c` configured.
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

void terminateHandler()
{
    int fd = g_logFd.load(std::memory_order_acquire);
    if(fd >= 0)
    {
        writeStr(fd, BC::CrashInternal::buildHeaderUtf8());
        writeStr(fd, "Unhandled C++ exception (std::terminate)\n");
        char metaBuf[128];
        std::snprintf(metaBuf, sizeof(metaBuf),
                      "Active experiment: %d\n\nStack trace:\n",
                      BC::CrashInternal::activeExperimentNumber());
        writeStr(fd, metaBuf);
        emitStackTrace(fd);
        ::fsync(fd);
    }
    std::abort();
}

}

namespace BC::CrashInternal {

void installPlatformHandlers()
{
    stack_t ss{};
    ss.ss_sp = g_altStack;
    ss.ss_size = kAltStackSize;
    ss.ss_flags = 0;
    ::sigaltstack(&ss, nullptr);

    struct sigaction sa{};
    sa.sa_sigaction = crashHandler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);

    for(int sig : { SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS })
        ::sigaction(sig, &sa, nullptr);

    std::set_terminate(terminateHandler);
}

void reopenPlatform()
{
    const char *dir = crashesDirUtf8();
    const char *base = currentBasenameUtf8();
    if(!dir || !*dir || !base || !*base)
        return;

    QByteArray path = QByteArray(dir) + base + ".log";

    int fd = ::open(path.constData(),
                    O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if(fd < 0)
        return;

    int oldFd = g_logFd.exchange(fd, std::memory_order_acq_rel);
    QByteArray oldPath = g_logPath;
    g_logPath = path;

    if(oldFd >= 0)
    {
        struct stat st{};
        bool empty = (::fstat(oldFd, &st) == 0 && st.st_size == 0);
        ::close(oldFd);
        if(empty && !oldPath.isEmpty())
            ::unlink(oldPath.constData());
    }
}

void shutdownPlatform()
{
    int fd = g_logFd.exchange(-1, std::memory_order_acq_rel);
    if(fd < 0)
        return;
    struct stat st{};
    bool empty = (::fstat(fd, &st) == 0 && st.st_size == 0);
    ::close(fd);
    if(empty && !g_logPath.isEmpty())
        ::unlink(g_logPath.constData());
    g_logPath.clear();
}

}
