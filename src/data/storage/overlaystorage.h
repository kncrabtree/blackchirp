#ifndef OVERLAYSTORAGE_H
#define OVERLAYSTORAGE_H

#include "datastoragebase.h"

#include <data/experiment/overlaybase.h>

namespace BC::Key::Overlay {
static const QString overlayDir{"overlays"};
static const QString overlayMdFile{"overlays.csv"};
}

class OverlayStorage : public DataStorageBase
{
public:
    OverlayStorage(int number, QString path);
    
    bool createOverlay(OverlayBase::OverlayType t, QString sourceFile, QString label = "");
    bool loadOverlay(QString fileBase, OverlayBase::OverlayType t);
    
    // DataStorageBase interface
    void advance() override {}
    void save() override;
    void start() override {}
    void finish() override {}
    
private:
    std::map<QString,OverlayBase*> d_overlays;
};

#endif // OVERLAYSTORAGE_H
