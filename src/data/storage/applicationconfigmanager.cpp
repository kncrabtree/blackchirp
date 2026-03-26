#include "applicationconfigmanager.h"

#include <QMutexLocker>
#include <QSettings>

// Initialize static member
ApplicationConfigManager* ApplicationConfigManager::s_instance = nullptr;

ApplicationConfigManager::ApplicationConfigManager(QObject *parent)
    : QObject(parent)
{
    initializeFromCompileTimeFlags();

    // Load persisted settings
    QSettings s;
    s.beginGroup(BC::Key::AppConfig::appConfig);
    d_currentConfig.debugLogging = s.value(BC::Key::AppConfig::debugLogging, true).toBool();
    s.endGroup();
}

ApplicationConfigManager& ApplicationConfigManager::instance()
{
    // Thread-safe singleton creation using Qt's implicit sharing
    static QMutex creationMutex;
    if (!s_instance) {
        QMutexLocker locker(&creationMutex);
        if (!s_instance) {
            s_instance = new ApplicationConfigManager();
        }
    }
    return *s_instance;
}

bool ApplicationConfigManager::isLifEnabled() const
{
    QMutexLocker locker(&d_configMutex);
    return d_currentConfig.lifEnabled;
}

bool ApplicationConfigManager::isCudaEnabled() const
{
    QMutexLocker locker(&d_configMutex);
    return d_currentConfig.cudaEnabled;
}

bool ApplicationConfigManager::isDebugLoggingEnabled() const
{
    QMutexLocker locker(&d_configMutex);
    return d_currentConfig.debugLogging;
}

void ApplicationConfigManager::setDebugLogging(bool enabled)
{
    {
        QMutexLocker locker(&d_configMutex);
        if(d_currentConfig.debugLogging == enabled)
            return;
        d_currentConfig.debugLogging = enabled;
    }

    QSettings s;
    s.beginGroup(BC::Key::AppConfig::appConfig);
    s.setValue(BC::Key::AppConfig::debugLogging, enabled);
    s.endGroup();

    emit debugLoggingChanged(enabled);
    emit configurationChanged(d_currentConfig);
}

void ApplicationConfigManager::initializeFromCompileTimeFlags()
{
    // Initialize configuration state from compile-time flags
    // This provides compatibility during the development transition phase

    d_currentConfig.lifEnabled = true;

    // Initialize CUDA state from BC_CUDA compilation flag
#ifdef BC_CUDA
    d_currentConfig.cudaEnabled = true;
#else
    d_currentConfig.cudaEnabled = false;
#endif
}
