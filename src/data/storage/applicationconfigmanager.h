#ifndef APPLICATIONCONFIGMANAGER_H
#define APPLICATIONCONFIGMANAGER_H

#include <QObject>
#include <QMutex>
#include <QVariant>
#include <QFont>
#include <QLatin1StringView>

/*!
 * \brief Declarative application option descriptor
 *
 * Describes a single configurable application option, including its settings
 * key, display name, description, type-aware default value, and whether
 * changing it requires an application restart.
 */
struct AppOption {
    QString settingsKey;       ///< QSettings key (within AppConfig group)
    QString label;             ///< Display name for UI
    QString description;       ///< Tooltip / help text
    QVariant defaultValue;     ///< Type-aware default
    bool requiresRestart;      ///< Show restart badge in UI
};

/*!
 * \brief Centralized runtime application configuration manager
 *
 * ApplicationConfigManager provides centralized, thread-safe runtime
 * configuration management. It replaces compile-time flags (BC_LIF, etc.)
 * with runtime configuration decisions persisted via QSettings.
 *
 * The manager maintains a declarative option registry (getOptions()) that
 * describes all available options with their metadata, enabling automatic
 * UI generation and generic get/set access.
 *
 * Design principles:
 * - Singleton pattern with global access
 * - Thread-safe with QMutex
 * - Qt integration with signals for configuration changes
 * - Declarative option registry for UI generation
 */
class ApplicationConfigManager : public QObject
{
    Q_OBJECT

public:
    /*!
     * \brief Application configuration structure
     */
    struct ApplicationConfig {
        bool lifEnabled{false};     /*!< LIF module enabled state */
        bool cudaEnabled{false};    /*!< CUDA module enabled state */
        bool debugLogging{false};   /*!< Debug log messages enabled state */
    };

    /*!
     * \brief Get singleton instance for application configuration access
     *
     * Provides thread-safe access to the application configuration manager.
     * The instance is created on first access and persists for the application
     * lifetime.
     *
     * \return Reference to the singleton instance
     */
    static ApplicationConfigManager& instance();

    /*!
     * \brief Check if LIF module is enabled
     *
     * Thread-safe query for LIF module availability.
     *
     * \return True if LIF functionality is enabled
     */
    bool isLifEnabled() const;

    /*!
     * \brief Check if CUDA module is enabled
     *
     * Thread-safe query for CUDA module availability.
     *
     * \return True if CUDA functionality is enabled
     */
    bool isCudaEnabled() const;

    /*!
     * \brief Check if debug logging is enabled
     * \return True if Debug-level log messages are written to the debug log file
     */
    bool isDebugLoggingEnabled() const;

    /*!
     * \brief Enable or disable debug logging and persist the setting
     * \param enabled True to write Debug messages to debug_YYYYMM.csv
     */
    void setDebugLogging(bool enabled);

    /*!
     * \brief Enable or disable LIF module and persist the setting
     * \param enabled True to enable LIF hardware and UI components
     */
    void setLifEnabled(bool enabled);

    /*!
     * \brief Get the declarative option registry
     * \return Read-only reference to the list of registered AppOption entries
     */
    const QVector<AppOption>& getOptions() const;

    /*!
     * \brief Get the current persisted value for an option key
     * \param key The settingsKey of the option (within AppConfig group)
     * \return The stored value, or the option's defaultValue if not yet set
     */
    QVariant getOptionValue(const QString& key) const;

    /*!
     * \brief Set and persist a value for an option key
     *
     * Persists the value to QSettings and updates in-memory state.
     * Emits the appropriate specific signal (lifEnabledChanged, fontChanged,
     * etc.) as well as configurationChanged when relevant.
     *
     * \param key The settingsKey of the option
     * \param value The new value to store
     */
    void setOptionValue(const QString& key, const QVariant& value);

signals:
    /*!
     * \brief Emitted when application configuration changes
     * \param newConfig The updated application configuration
     */
    void configurationChanged(const ApplicationConfig& newConfig);
    void debugLoggingChanged(bool enabled);
    void fontChanged(QFont font);
    void lifEnabledChanged(bool enabled);

private:
    explicit ApplicationConfigManager(QObject *parent = nullptr);
    ~ApplicationConfigManager() = default;

    ApplicationConfigManager(const ApplicationConfigManager&) = delete;
    ApplicationConfigManager& operator=(const ApplicationConfigManager&) = delete;

    static ApplicationConfigManager* s_instance;  /*!< Singleton instance */
    mutable QMutex d_configMutex;                /*!< Thread-safe access control */
    ApplicationConfig d_currentConfig;            /*!< Current configuration state */
    QVector<AppOption> d_options;                 /*!< Declarative option registry */
};

/*!
 * \brief Settings keys for application configuration
 */
namespace BC::Key::AppConfig {
    inline constexpr QLatin1StringView appConfig{"applicationConfig"};     /*!< Base settings key */
    inline constexpr QLatin1StringView lifEnabled{"lifEnabled"};           /*!< LIF enabled state */
    inline constexpr QLatin1StringView cudaEnabled{"cudaEnabled"};         /*!< CUDA enabled state */
    inline constexpr QLatin1StringView debugLogging{"debugLogging"};       /*!< Debug logging enabled state */
    inline constexpr QLatin1StringView appFont{"appFont"};                 /*!< Application font */
}

#endif // APPLICATIONCONFIGMANAGER_H
