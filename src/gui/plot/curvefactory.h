#ifndef CURVEFACTORY_H
#define CURVEFACTORY_H

#include <memory>
#include <map>
#include <QString>
#include <QVariant>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>

#include <data/storage/settingsstorage.h>

class OverlayBase;

/*!
 * \brief Abstract interface for curve display settings storage
 * 
 * This interface allows curve classes to store their display settings
 * in different backends (QSettings via SettingsStorage, or overlay metadata).
 */
class CurveStorageInterface {
public:
    virtual ~CurveStorageInterface() = default;
    
    /*!
     * \brief Store a value with the given key
     * \param key The settings key
     * \param value The value to store
     */
    virtual void set(const QString& key, const QVariant& value) = 0;
    
    /*!
     * \brief Retrieve a value for the given key
     * \param key The settings key
     * \param defaultValue Default value if key doesn't exist
     * \return The stored value or default
     */
    virtual QVariant get(const QString& key, const QVariant& defaultValue = QVariant()) const = 0;
    
    /*!
     * \brief Type-safe template method for retrieving values
     * \param key The settings key
     * \param defaultValue Default value if key doesn't exist
     * \return The stored value or default, cast to type T
     */
    template<typename T>
    inline T get(const QString key, const T &defaultValue = QVariant().value<T>()) const { 
        QVariant val = get(key, QVariant::fromValue(defaultValue));
        return val.value<T>(); 
    }
    
    /*!
     * \brief Type-safe template method for storing values
     * \param key The settings key
     * \param value The value to store
     */
    template<typename T>
    void set(const QString& key, const T& value) {
        set(key, QVariant::fromValue(value));
    }
};

/*!
 * \brief SettingsStorage wrapper implementing CurveStorageInterface
 * 
 * This wrapper allows existing SettingsStorage-based curves to use
 * the new storage interface while maintaining backward compatibility.
 */
class SettingsStorageWrapper : public CurveStorageInterface, public SettingsStorage {
public:
    /*!
     * \brief Constructor mirroring BlackchirpPlotCurveBase's SettingsStorage initialization
     * \param key The settings key for this curve
     * \param category The settings category (default: General)
     */
    SettingsStorageWrapper(const QString& key, SettingsStorage::Type type = SettingsStorage::General);
    
    // CurveStorageInterface implementation
    void set(const QString& key, const QVariant& value) override;
    QVariant get(const QString& key, const QVariant& defaultValue) const override;
};

/*!
 * \brief Overlay metadata storage implementing CurveStorageInterface
 * 
 * This storage backend stores curve display settings as metadata
 * within the OverlayBase object, allowing overlay-specific persistence.
 */
class OverlayMetadataStorage : public CurveStorageInterface {
public:
    /*!
     * \brief Constructor
     * \param overlay The overlay object to store metadata in
     */
    OverlayMetadataStorage(OverlayBase* overlay);
    
    // CurveStorageInterface implementation
    void set(const QString& key, const QVariant& value) override;
    QVariant get(const QString& key, const QVariant& defaultValue) const override;
    
    /*!
     * \brief Synchronize settings from overlay metadata to cache
     */
    void syncFromOverlay();
    
    /*!
     * \brief Synchronize settings from cache to overlay metadata
     */
    void syncToOverlay();
    
private:
    OverlayBase* d_overlay;
    mutable std::map<QString, QVariant> d_cache;
};

/*!
 * \brief Factory class for creating curve objects with appropriate storage backends
 * 
 * This factory provides templated methods to create curve objects with either
 * SettingsStorage (for standard curves) or OverlayMetadata (for overlay curves)
 * storage backends.
 */
class CurveFactory {
public:
    /*!
     * \brief Create a standard curve with SettingsStorage backend
     * \param key The settings key for the curve
     * \param category The settings category (default: General)
     * \param title The curve title (default: empty)
     * \param defaultLineStyle Default line style (default: SolidLine)
     * \param defaultMarker Default marker style (default: NoSymbol)
     * \param defaultStyle Default curve style (default: Lines)
     * \return Unique pointer to the created curve
     */
    template<typename CurveType>
    static std::unique_ptr<CurveType> 
    createStandardCurve(const QString& key, 
                       SettingsStorage::Type type = SettingsStorage::General,
                       const QString& title = QString(""),
                       Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                       QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                       QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines) {
        
        auto storage = std::make_unique<SettingsStorageWrapper>(key, type);
        return std::make_unique<CurveType>(std::move(storage), key, title, 
                                          defaultLineStyle, defaultMarker, defaultStyle);
    }
    
    /*!
     * \brief Create an overlay curve with OverlayMetadata backend
     * \param key The settings key for the curve
     * \param overlay The overlay object to store metadata in
     * \param title The curve title (default: empty)
     * \param defaultLineStyle Default line style (default: SolidLine)
     * \param defaultMarker Default marker style (default: NoSymbol)
     * \param defaultStyle Default curve style (default: Lines)
     * \return Unique pointer to the created curve
     */
    template<typename CurveType>
    static std::unique_ptr<CurveType> 
    createOverlayCurve(const QString& key, 
                      OverlayBase* overlay,
                      const QString& title = QString(""),
                      Qt::PenStyle defaultLineStyle = Qt::SolidLine,
                      QwtSymbol::Style defaultMarker = QwtSymbol::NoSymbol,
                      QwtPlotCurve::CurveStyle defaultStyle = QwtPlotCurve::Lines) {
        
        auto storage = std::make_unique<OverlayMetadataStorage>(overlay);
        return std::make_unique<CurveType>(std::move(storage), key, title, 
                                          defaultLineStyle, defaultMarker, defaultStyle);
    }
};

#endif // CURVEFACTORY_H