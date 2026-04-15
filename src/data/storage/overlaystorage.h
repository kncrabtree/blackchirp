#ifndef OVERLAYSTORAGE_H
#define OVERLAYSTORAGE_H

#include "datastoragebase.h"

#include <data/experiment/overlaybase.h>
#include <memory>
#include <QVector>
#include <QFuture>
#include <QObject>
#include <QLatin1StringView>

namespace BC::Key::Overlay {
inline constexpr QLatin1StringView overlayDir{"overlays"};
inline constexpr QLatin1StringView overlayMdFile{"overlays.csv"};
inline constexpr QLatin1StringView bcMajorVersion{"BCMajorVersion"};
inline constexpr QLatin1StringView bcMinorVersion{"BCMinorVersion"};
inline constexpr QLatin1StringView bcPatchVersion{"BCPatchVersion"};
inline constexpr QLatin1StringView bcReleaseVersion{"BCReleaseVersion"};
inline constexpr QLatin1StringView bcBuildVersion{"BCBuildVersion"};
}

class OverlayStorage : public QObject, public DataStorageBase
{
    Q_OBJECT
    
public:
    OverlayStorage(int number, QString path);
    ~OverlayStorage();
    
    bool loadOverlay(QString fileBase, OverlayBase::OverlayType t);
    
    // Add externally created overlay to storage
    bool addOverlay(std::shared_ptr<OverlayBase> overlay);
    
    // Get all overlays as shared_ptr for safe external use
    QVector<std::shared_ptr<OverlayBase>> getAllOverlays() const;
    
    // Remove overlay by label
    bool removeOverlay(const QString& label);
    
    // Rename overlay and associated files
    bool renameOverlay(const QString& currentLabel, const QString& newLabel);
    
    // Async write management
    bool hasPendingWrites() const;
    void waitForPendingWrites();
    int pendingWriteCount() const;
    
    // Save only the metadata for a specific overlay (used when curve settings change)
    void saveOverlayMetadata(std::shared_ptr<OverlayBase> overlay);
    
    // Preview overlay management (temporary overlays not persisted to disk)
    bool addPreviewOverlay(std::shared_ptr<OverlayBase> overlay);
    bool removePreviewOverlay(const QString& label);
    bool detachPreviewOverlay(const QString& label);
    void clearAllPreviews();
    QVector<std::shared_ptr<OverlayBase>> getAllPreviewOverlays() const;
    
    // DataStorageBase interface
    void advance() override {}
    void save() override;
    void start() override {}
    void finish() override {}
    
signals:
    void overlayAdded(std::shared_ptr<OverlayBase> overlay);
    void overlayRemoved(std::shared_ptr<OverlayBase> overlay);
    void overlayWriteCompleted(std::shared_ptr<OverlayBase> overlay);
    void overlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error);
    void pendingWritesChanged(int count);
    
private:
    std::map<QString, std::shared_ptr<OverlayBase>> d_overlays;
    std::map<QString, QFuture<void>> d_pendingWrites; // Maps overlay label to write future
    
    // Preview overlays (temporary, not persisted)
    std::map<QString, std::shared_ptr<OverlayBase>> d_previewOverlays;
    
    // Factory method for creating overlay objects
    std::shared_ptr<OverlayBase> createOverlayObject(OverlayBase::OverlayType type);
    
    // Helper methods
    QString sanitizeLabel(const QString& label) const;
    QString getOverlayDataPath(const QString& sanitizedLabel) const;
    QString getOverlaySettingsPath(const QString& sanitizedLabel) const;
    void addVersionMetadata(std::map<QString, QVariant>& metadata) const;
    bool validateOverlayLabel(const QString& label) const;
    void onWriteCompleted(const QString& label, bool success, const QString& error = QString());
};

#endif // OVERLAYSTORAGE_H
