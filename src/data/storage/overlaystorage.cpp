#include "overlaystorage.h"


OverlayStorage::OverlayStorage(int number, QString path) :
    DataStorageBase(number,path)
{
    //look for overlays.csv file, if found, load overlays
    std::map<QString,QVariant> m;
    readMetadata(BC::Key::Overlay::overlayMdFile,m,BC::Key::Overlay::overlayDir);
    for(auto it = m.cbegin(); it != m.cend(); ++it)
    {
        auto fileBase = it->first;
        auto type = it->second.value<OverlayBase::OverlayType>();
        
        loadOverlay(fileBase,type);           
    }
}

bool OverlayStorage::createOverlay(OverlayBase::OverlayType t, QString sourceFile, QString label)
{
    auto it = d_overlays.find(label);
    if(it != d_overlays.end())
        return false;
    
    //create overlay, add to map
    
    return true;
}

bool OverlayStorage::loadOverlay(QString fileBase, OverlayBase::OverlayType t)
{   
    //read in overlay, add to map
    
    return true;
}

void OverlayStorage::save()
{
    if(d_number < 1)
        return;
    
    using namespace BC::Key::Overlay;
    std::map<QString,QVariant> m;
    
    for(auto it = d_overlays.begin(); it != d_overlays.end(); ++it)
    {
        m.emplace(it->first,it->second->type());
        std::map<QString,QVariant> overlaySettings;
        it->second->storeMetadata(overlaySettings);
        writeMetadata(overlaySettingsFile.arg(it->first),overlaySettings,overlayDir);
        it->second->save();
    }
    
    writeMetadata(overlayMdFile,m,overlayDir);
    
}
