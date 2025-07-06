#ifndef OVERLAYSTORAGE_H
#define OVERLAYSTORAGE_H

#include "datastoragebase.h"

#include <data/experiment/overlaybase.h>
#include <memory>
#include <QVector>

namespace BC::Key::Overlay {
static const QString overlayDir{"overlays"};
static const QString overlayMdFile{"overlays.csv"};
static const QString bcMajorVersion{"BCMajorVersion"};
static const QString bcMinorVersion{"BCMinorVersion"};
static const QString bcPatchVersion{"BCPatchVersion"};
static const QString bcReleaseVersion{"BCReleaseVersion"};
static const QString bcBuildVersion{"BCBuildVersion"};
}

class OverlayStorage : public DataStorageBase
{
public:
    OverlayStorage(int number, QString path);
    ~OverlayStorage();
    
    bool createOverlay(OverlayBase::OverlayType t, QString sourceFile, QString label = "");
    bool loadOverlay(QString fileBase, OverlayBase::OverlayType t);
    
    // Add externally created overlay to storage
    bool addOverlay(std::shared_ptr<OverlayBase> overlay);
    
    // Get all overlays as shared_ptr for safe external use
    QVector<std::shared_ptr<OverlayBase>> getAllOverlays() const;
    
    // Remove overlay by label
    bool removeOverlay(const QString& label);
    
    // DataStorageBase interface
    void advance() override {}
    void save() override;
    void start() override {}
    void finish() override {}
    
private:
    std::map<QString, std::shared_ptr<OverlayBase>> d_overlays;
    
    // Factory method for creating overlay objects
    std::shared_ptr<OverlayBase> createOverlayObject(OverlayBase::OverlayType type);
    
    // Helper methods
    QString sanitizeLabel(const QString& label) const;
    void addVersionMetadata(std::map<QString, QVariant>& metadata) const;
    bool validateOverlayLabel(const QString& label) const;
};

#endif // OVERLAYSTORAGE_H
