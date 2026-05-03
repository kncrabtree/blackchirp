#ifndef CURVEFACTORY_H
#define CURVEFACTORY_H

#include <memory>
#include <QString>
#include <QVariant>
#include <qwt6/qwt_plot_curve.h>
#include <qwt6/qwt_symbol.h>

#include <data/storage/settingsstorage.h>

class OverlayBase;

/// \brief Polymorphic key/value storage contract for curve display settings.
///
/// \sa SettingsStorageWrapper, OverlayMetadataStorage, BlackchirpPlotCurveBase
class CurveStorageInterface {
public:
    /// \brief Virtual destructor.
    virtual ~CurveStorageInterface() = default;

    /// \brief Stores \a value under \a key.
    /// \param key   Settings key.
    /// \param value Value to store.
    virtual void set(const QString& key, const QVariant& value) = 0;

    /// \brief Retrieves the value stored under \a key.
    /// \param key          Settings key.
    /// \param defaultValue Returned when \a key is absent.
    /// \return Stored value, or \a defaultValue.
    virtual QVariant get(const QString& key, const QVariant& defaultValue = QVariant()) const = 0;

    /// \brief Type-safe retrieval.
    /// \tparam T Expected value type.
    /// \param key          Settings key.
    /// \param defaultValue Returned when \a key is absent.
    /// \return Stored value cast to \c T, or \a defaultValue.
    template<typename T>
    inline T get(const QString key, const T &defaultValue = QVariant().value<T>()) const {
        QVariant val = get(key, QVariant::fromValue(defaultValue));
        return val.value<T>();
    }

    /// \brief Type-safe storage.
    /// \tparam T Value type; must be registered with QMetaType.
    /// \param key   Settings key.
    /// \param value Value to store.
    template<typename T>
    void set(const QString& key, const T& value) {
        set(key, QVariant::fromValue(value));
    }
};

/// \brief CurveStorageInterface backed by SettingsStorage (QSettings).
///
/// Adapts the SettingsStorage / QSettings persistence layer to the
/// CurveStorageInterface contract. Standard (non-overlay) curves use
/// this backend so their appearance survives application restarts.
///
/// \sa CurveStorageInterface, SettingsStorage, CurveFactory::createStandardCurve
class SettingsStorageWrapper : public CurveStorageInterface, public SettingsStorage {
public:
    /// \brief Constructs the wrapper with the given key and storage type.
    /// \param key  Settings key identifying this curve's storage group.
    /// \param type Storage category (default: SettingsStorage::General).
    SettingsStorageWrapper(const QString& key, SettingsStorage::Type type = SettingsStorage::General);

    /// \brief Stores \a value under \a key via SettingsStorage.
    void set(const QString& key, const QVariant& value) override;

    /// \brief Retrieves the value stored under \a key from SettingsStorage.
    QVariant get(const QString& key, const QVariant& defaultValue) const override;
};

/// \brief CurveStorageInterface backed by an OverlayBase metadata blob.
///
/// Routes all \c set / \c get calls into the metadata map of the supplied
/// OverlayBase. Overlay curves use this backend so their appearance is
/// saved alongside the overlay data rather than in QSettings.
///
/// \sa CurveStorageInterface, OverlayBase, CurveFactory::createOverlayCurve
class OverlayMetadataStorage : public CurveStorageInterface {
public:
    /// \brief Constructs storage that writes into \a overlay's metadata.
    /// \param overlay Overlay whose metadata blob receives the curve settings.
    OverlayMetadataStorage(std::shared_ptr<OverlayBase> overlay);

    /// \brief Stores \a value under \a key in the overlay metadata.
    void set(const QString& key, const QVariant& value) override;

    /// \brief Retrieves the value stored under \a key from the overlay metadata.
    QVariant get(const QString& key, const QVariant& defaultValue) const override;

    /// \brief Returns the overlay this storage writes into.
    std::shared_ptr<OverlayBase> getOverlay() const { return d_overlay; }

private:
    std::shared_ptr<OverlayBase> d_overlay;
};

/// \brief Factory that constructs BlackchirpPlotCurveBase subclasses with the
///        correct storage backend.
///
/// \sa CurveStorageInterface, BlackchirpPlotCurveBase, OverlayBase
class CurveFactory {
public:
    /// \brief Creates a curve with a SettingsStorageWrapper backend.
    ///
    /// \tparam CurveType Concrete subclass of BlackchirpPlotCurveBase to
    ///         instantiate.
    /// \param key                Settings key for this curve.
    /// \param type               Storage category (default: SettingsStorage::General).
    /// \param title              Curve title shown in the legend (default: empty).
    /// \param defaultLineStyle   Initial line style (default: Qt::SolidLine).
    /// \param defaultMarker      Initial symbol style (default: QwtSymbol::NoSymbol).
    /// \param defaultStyle       Initial curve style (default: QwtPlotCurve::Lines).
    /// \return Owning pointer to the constructed curve.
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

    /// \brief Creates a curve with an OverlayMetadataStorage backend.
    ///
    /// \tparam CurveType Concrete subclass of BlackchirpPlotCurveBase to
    ///         instantiate.
    /// \param key                Settings key for this curve.
    /// \param overlay            Overlay that will own the appearance metadata.
    /// \param title              Curve title shown in the legend (default: empty).
    /// \param defaultLineStyle   Initial line style (default: Qt::SolidLine).
    /// \param defaultMarker      Initial symbol style (default: QwtSymbol::NoSymbol).
    /// \param defaultStyle       Initial curve style (default: QwtPlotCurve::Lines).
    /// \return Owning pointer to the constructed curve.
    template<typename CurveType>
    static std::unique_ptr<CurveType>
    createOverlayCurve(const QString& key,
                      std::shared_ptr<OverlayBase> overlay,
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
