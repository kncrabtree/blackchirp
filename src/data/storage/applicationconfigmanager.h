#ifndef APPLICATIONCONFIGMANAGER_H
#define APPLICATIONCONFIGMANAGER_H

#include <QObject>
#include <QMutex>

/*!
 * \brief Centralized application state management for feature enabling
 * 
 * ApplicationConfigManager provides centralized, thread-safe runtime configuration
 * management to replace compilation flags with runtime configuration decisions.
 * 
 * This class implements the singleton pattern with thread-safe access to application
 * feature state. During development, the configuration is initialized from existing
 * compilation flags (BC_LIF, BC_CUDA) to maintain compatibility while transitioning
 * from compile-time to runtime configuration.
 * 
 * Design principles:
 * - Singleton pattern with global access
 * - Thread-safe with QMutex (simple, non-over-engineered)
 * - Compile-time flag initialization during development phase
 * - Qt integration with signals for configuration changes
 * - Simple, straightforward implementation
 * 
 * Usage example:
 * ```cpp
 * // Check if LIF is enabled (replaces #ifdef BC_LIF)
 * if (ApplicationConfigManager::instance().isLifEnabled()) {
 *     // LIF-specific code
 * }
 * 
 * // Check if CUDA is enabled (replaces #ifdef BC_CUDA)
 * if (ApplicationConfigManager::instance().isCudaEnabled()) {
 *     // CUDA-specific code
 * }
 * ```
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
     * Thread-safe query for LIF module availability. This replaces compile-time
     * #ifdef BC_LIF checks with runtime configuration queries.
     * 
     * \return True if LIF functionality is enabled
     */
    bool isLifEnabled() const;

    /*!
     * \brief Check if CUDA module is enabled
     * 
     * Thread-safe query for CUDA module availability. This replaces compile-time
     * #ifdef BC_CUDA checks with runtime configuration queries.
     * 
     * \return True if CUDA functionality is enabled
     */
    bool isCudaEnabled() const;

signals:
    /*!
     * \brief Emitted when application configuration changes
     * 
     * This signal is emitted whenever the application configuration is modified,
     * allowing components to respond to configuration changes dynamically.
     * 
     * \param newConfig The updated application configuration
     */
    void configurationChanged(const ApplicationConfig& newConfig);

private:
    /*!
     * \brief Private constructor for singleton pattern
     * 
     * Initializes the application configuration from compile-time flags during
     * the development phase. This ensures compatibility while transitioning from
     * compile-time to runtime configuration.
     */
    explicit ApplicationConfigManager(QObject *parent = nullptr);

    /*!
     * \brief Private destructor
     */
    ~ApplicationConfigManager() = default;

    // Disable copy/assignment for singleton pattern
    ApplicationConfigManager(const ApplicationConfigManager&) = delete;
    ApplicationConfigManager& operator=(const ApplicationConfigManager&) = delete;

    /*!
     * \brief Initialize configuration from compile-time flags
     * 
     * During the development phase, this method initializes the runtime
     * configuration state based on existing BC_LIF and BC_CUDA compilation
     * flags. This provides a smooth transition from compile-time to runtime
     * configuration management.
     * 
     * This method is called once during singleton construction.
     */
    void initializeFromCompileTimeFlags();

    static ApplicationConfigManager* s_instance;  /*!< Singleton instance */
    mutable QMutex d_configMutex;                /*!< Thread-safe access control */
    ApplicationConfig d_currentConfig;            /*!< Current configuration state */
};

/*!
 * \brief Settings keys for application configuration
 */
namespace BC::Key::AppConfig {
    static const QString appConfig{"applicationConfig"};     /*!< Base settings key */
    static const QString lifEnabled{"lifEnabled"};           /*!< LIF enabled state */
    static const QString cudaEnabled{"cudaEnabled"};         /*!< CUDA enabled state */
}

#endif // APPLICATIONCONFIGMANAGER_H