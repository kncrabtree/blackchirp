#include "applicationconfigmanager.h"

#include <QFont>
#include <QMutexLocker>
#include <QSettings>

#include "settingsstorage.h"

// Initialize static member
ApplicationConfigManager* ApplicationConfigManager::s_instance = nullptr;

ApplicationConfigManager::ApplicationConfigManager(QObject *parent)
    : QObject(parent)
{
    // Register declarative option entries
    d_options.append({
        BC::Key::AppConfig::lifEnabled,
        QStringLiteral("LIF Module"),
        QStringLiteral("Enable Laser-Induced Fluorescence hardware and UI components"),
        QVariant(true),
        true
    });
    d_options.append({
        BC::Key::AppConfig::debugLogging,
        QStringLiteral("Debug Logging"),
        QStringLiteral("Write debug-level messages to the log file"),
        QVariant(false),
        false
    });
    d_options.append({
        BC::Key::AppConfig::appFont,
        QStringLiteral("Application Font"),
        QStringLiteral("Font used throughout the application"),
        QVariant::fromValue(QFont(QStringLiteral("sans-serif"), 8)),
        false
    });

    // Load persisted settings
    QSettings s;
    s.beginGroup(BC::Key::AppConfig::appConfig);
    d_currentConfig.lifEnabled = s.value(BC::Key::AppConfig::lifEnabled, true).toBool();
    d_currentConfig.debugLogging = s.value(BC::Key::AppConfig::debugLogging, false).toBool();
    s.endGroup();

    // Font migration: copy from old location if new key is absent
    {
        QSettings ms;
        ms.beginGroup(BC::Key::AppConfig::appConfig);
        bool hasNewFont = ms.contains(BC::Key::AppConfig::appFont);
        ms.endGroup();

        if(!hasNewFont)
        {
            QSettings old;
            old.beginGroup(BC::Key::BC);
            if(old.contains(BC::Key::appFont))
            {
                QVariant fontVal = old.value(BC::Key::appFont);
                old.remove(BC::Key::appFont);
                old.endGroup();

                QSettings ns;
                ns.beginGroup(BC::Key::AppConfig::appConfig);
                ns.setValue(BC::Key::AppConfig::appFont, fontVal);
                ns.endGroup();
            }
            else
            {
                old.endGroup();
            }
        }
    }

    // Initialize CUDA state from BC_CUDA compilation flag
#ifdef BC_CUDA
    d_currentConfig.cudaEnabled = true;
#endif
}

ApplicationConfigManager& ApplicationConfigManager::instance()
{
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

void ApplicationConfigManager::setLifEnabled(bool enabled)
{
    {
        QMutexLocker locker(&d_configMutex);
        if(d_currentConfig.lifEnabled == enabled)
            return;
        d_currentConfig.lifEnabled = enabled;
    }

    QSettings s;
    s.beginGroup(BC::Key::AppConfig::appConfig);
    s.setValue(BC::Key::AppConfig::lifEnabled, enabled);
    s.endGroup();

    emit lifEnabledChanged(enabled);
    emit configurationChanged(d_currentConfig);
}

const QVector<AppOption>& ApplicationConfigManager::getOptions() const
{
    return d_options;
}

QVariant ApplicationConfigManager::getOptionValue(const QString& key) const
{
    QSettings s;
    s.beginGroup(BC::Key::AppConfig::appConfig);

    for(const auto& opt : d_options)
    {
        if(opt.settingsKey == key)
        {
            auto val = s.value(key, opt.defaultValue);
            s.endGroup();
            return val;
        }
    }

    s.endGroup();
    return {};
}

void ApplicationConfigManager::setOptionValue(const QString& key, const QVariant& value)
{
    QSettings s;
    s.beginGroup(BC::Key::AppConfig::appConfig);
    s.setValue(key, value);
    s.endGroup();

    if(key == BC::Key::AppConfig::lifEnabled)
    {
        bool enabled = value.toBool();
        {
            QMutexLocker locker(&d_configMutex);
            if(d_currentConfig.lifEnabled == enabled)
                return;
            d_currentConfig.lifEnabled = enabled;
        }
        emit lifEnabledChanged(enabled);
        emit configurationChanged(d_currentConfig);
    }
    else if(key == BC::Key::AppConfig::debugLogging)
    {
        setDebugLogging(value.toBool());
    }
    else if(key == BC::Key::AppConfig::appFont)
    {
        emit fontChanged(value.value<QFont>());
    }
}
