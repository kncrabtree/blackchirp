#include "applicationconfigmanager.h"

#include <QMutexLocker>

// Initialize static member
ApplicationConfigManager* ApplicationConfigManager::s_instance = nullptr;

ApplicationConfigManager::ApplicationConfigManager(QObject *parent)
    : QObject(parent)
{
    // Initialize configuration from compile-time flags during development
    initializeFromCompileTimeFlags();
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
