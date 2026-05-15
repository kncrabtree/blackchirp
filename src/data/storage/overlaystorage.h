#ifndef OVERLAYSTORAGE_H
#define OVERLAYSTORAGE_H

#include "datastoragebase.h"

#include <data/experiment/overlaybase.h>
#include <memory>
#include <QVector>
#include <QFuture>
#include <QObject>
#include <QLatin1StringView>

/// \brief Additional storage keys used by OverlayStorage.
namespace BC::Key::Overlay {
inline constexpr QLatin1StringView overlayDir{"overlays"};           ///< Subdirectory name under the experiment path.
inline constexpr QLatin1StringView overlayMdFile{"overlays.csv"};    ///< Filename of the overlay metadata index.
inline constexpr QLatin1StringView bcMajorVersion{"BCMajorVersion"}; ///< Blackchirp major version recorded in the overlay index.
inline constexpr QLatin1StringView bcMinorVersion{"BCMinorVersion"}; ///< Blackchirp minor version recorded in the overlay index.
inline constexpr QLatin1StringView bcPatchVersion{"BCPatchVersion"}; ///< Blackchirp patch version recorded in the overlay index.
inline constexpr QLatin1StringView bcReleaseVersion{"BCReleaseVersion"}; ///< Blackchirp release label recorded in the overlay index.
inline constexpr QLatin1StringView bcBuildVersion{"BCBuildVersion"}; ///< Blackchirp build number recorded in the overlay index.
}

/*!
 * \brief Manages the persistent collection of overlays for a single experiment.
 *
 * OverlayStorage maintains two separate collections of OverlayBase objects:
 * a *persistent* map of overlays that are written to disk under
 * \c \<experimentPath\>/overlays/, and a *preview* map of temporary overlays
 * that exist only in memory.  Each persistent overlay occupies two files: a
 * data file and a settings CSV, both named from the sanitized overlay label.
 *
 * Write operations are dispatched asynchronously via QtConcurrent so that
 * the calling thread is not blocked during I/O.  The signals
 * overlayWriteCompleted() and overlayWriteFailed() report completion on the
 * object's thread.  hasPendingWrites() and waitForPendingWrites() let callers
 * coordinate with the background tasks when needed (e.g. before closing the
 * experiment).
 *
 * Preview overlays bypass all disk I/O and are managed separately through
 * addPreviewOverlay(), removePreviewOverlay(), detachPreviewOverlay(), and
 * clearAllPreviews().  Detaching converts a preview overlay to a persistent
 * one by clearing its preview flag and adding it to the persistent map.
 *
 * OverlayStorage inherits DataStorageBase for interface compatibility with the
 * experiment data pipeline; only save() has a non-trivial implementation —
 * advance(), start(), and finish() are no-ops.
 *
 * \sa OverlayBase, DataStorageBase
 */
class OverlayStorage : public QObject, public DataStorageBase
{
    Q_OBJECT

public:
    /*!
     * \brief Construct the storage manager for a specific experiment.
     * \param number Experiment number; used to locate the experiment directory.
     * \param path   Base path of the experiment data directory.
     */
    OverlayStorage(int number, const QString &path);
    ~OverlayStorage();

    /*!
     * \brief Load an existing overlay from disk into the persistent collection.
     *
     * Constructs the appropriate OverlayBase subclass for \p t, sets its
     * source and destination paths from \p fileBase, reads the settings CSV,
     * and calls readFromDest() to populate its data.
     * \param fileBase Sanitized base name of the overlay files (without extension).
     * \param t        OverlayType discriminator identifying the subclass to create.
     * \return \c true if the overlay was loaded and added successfully.
     */
    bool loadOverlay(const QString &fileBase, OverlayBase::OverlayType t);

    /*!
     * \brief Add an externally created overlay to the persistent collection.
     *
     * Schedules an asynchronous write of the overlay data and settings.
     * Emits overlayAdded() on success.
     * \param overlay Shared pointer to the overlay to store.
     * \return \c true if the overlay was accepted (label is unique and valid).
     */
    bool addOverlay(std::shared_ptr<OverlayBase> overlay);

    /*!
     * \brief Return all persistent overlays as a vector of shared pointers.
     */
    QVector<std::shared_ptr<OverlayBase>> getAllOverlays() const;

    /*!
     * \brief Remove the persistent overlay with the given label and delete its files.
     *
     * Emits overlayRemoved() if the overlay existed.
     * \param label User-visible label of the overlay to remove.
     * \return \c true if the overlay was found and removed.
     */
    bool removeOverlay(const QString& label);

    /*!
     * \brief Rename a persistent overlay and its associated files.
     * \param currentLabel Existing label.
     * \param newLabel     New label; must be unique and valid.
     * \return \c true if the rename succeeded.
     */
    bool renameOverlay(const QString& currentLabel, const QString& newLabel);

    /*!
     * \brief Return \c true if any background write operations are still in flight.
     */
    bool hasPendingWrites() const;

    /*!
     * \brief Block until all pending background write operations complete.
     */
    void waitForPendingWrites();

    /*!
     * \brief Return the number of write operations currently in flight.
     */
    int pendingWriteCount() const;

    /*!
     * \brief Asynchronously write only the settings metadata for a specific overlay.
     *
     * Used when curve appearance or other non-data settings change without
     * requiring a full data rewrite.
     * \param overlay Overlay whose settings should be saved.
     */
    void saveOverlayMetadata(std::shared_ptr<OverlayBase> overlay);

    /*!
     * \brief Add a preview (non-persistent) overlay to the preview collection.
     * \param overlay Shared pointer to the preview overlay.
     * \return \c true if the overlay was accepted.
     */
    bool addPreviewOverlay(std::shared_ptr<OverlayBase> overlay);

    /*!
     * \brief Remove a preview overlay by label.
     * \param label Label of the preview overlay to remove.
     * \return \c true if the overlay was found and removed.
     */
    bool removePreviewOverlay(const QString& label);

    /*!
     * \brief Convert a preview overlay to a persistent overlay.
     *
     * Clears the preview flag on the overlay and moves it from the preview
     * collection to the persistent collection, scheduling a background write.
     * \param label Label of the preview overlay to detach.
     * \return \c true if the overlay was found and detached successfully.
     */
    bool detachPreviewOverlay(const QString& label);

    /*!
     * \brief Detach a preview overlay identified by object identity.
     *
     * Equivalent to detachPreviewOverlay(const QString&) but matches the
     * preview entry by shared_ptr rather than label. The label-keyed form
     * misses when the overlay's label changed after it was registered as a
     * preview, which then lets clearAllPreviews() emit overlayRemoved for the
     * just-promoted overlay and tear its curve back off the plot.
     * \param overlay The preview overlay instance to detach.
     * \return \c true if a matching preview entry was found and removed.
     */
    bool detachPreviewOverlay(const std::shared_ptr<OverlayBase>& overlay);

    /// \brief Remove all preview overlays from the preview collection.
    void clearAllPreviews();

    /*!
     * \brief Return all preview overlays as a vector of shared pointers.
     */
    QVector<std::shared_ptr<OverlayBase>> getAllPreviewOverlays() const;

    // DataStorageBase interface
    /// \brief No-op; OverlayStorage does not use segment-advance semantics.
    void advance() override {}
    /*!
     * \brief Write the overlay metadata index (overlays.csv) to disk.
     */
    void save() override;
    /// \brief No-op; OverlayStorage does not require explicit start.
    void start() override {}
    /// \brief No-op; OverlayStorage does not require explicit finish.
    void finish() override {}

signals:
    /// \brief Emitted when an overlay is successfully added to the persistent collection.
    void overlayAdded(std::shared_ptr<OverlayBase> overlay);
    /// \brief Emitted when an overlay is removed from the persistent collection.
    void overlayRemoved(std::shared_ptr<OverlayBase> overlay);
    /// \brief Emitted when a background write operation completes successfully.
    void overlayWriteCompleted(std::shared_ptr<OverlayBase> overlay);
    /// \brief Emitted when a background write operation fails.
    void overlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error);
    /// \brief Emitted when the number of in-flight write operations changes.
    void pendingWritesChanged(int count);

private:
    std::map<QString, std::shared_ptr<OverlayBase>, std::less<>> d_overlays;
    std::map<QString, QFuture<void>, std::less<>> d_pendingWrites;

    // Preview overlays (temporary, not persisted)
    std::map<QString, std::shared_ptr<OverlayBase>, std::less<>> d_previewOverlays;

    // Factory method for creating overlay objects
    std::shared_ptr<OverlayBase> createOverlayObject(OverlayBase::OverlayType type);

    // Helper methods
    QString sanitizeLabel(const QString& label) const;
    QString getOverlayDataPath(const QString& sanitizedLabel) const;
    QString getOverlaySettingsPath(const QString& sanitizedLabel) const;
    void addVersionMetadata(std::map<QString, QVariant, std::less<>>& metadata) const;
    bool validateOverlayLabel(const QString& label) const;
    void onWriteCompleted(const QString& label, bool success, const QString& error = QString());
};

#endif // OVERLAYSTORAGE_H
