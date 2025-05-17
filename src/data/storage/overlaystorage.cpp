#include "overlaystorage.h"


OverlayStorage::OverlayStorage(int number, QString path) :
    DataStorageBase(number,path)
{
    
}


void OverlayStorage::advance()
{
}

void OverlayStorage::save()
{
    if(d_number < 1)
        return;
}

void OverlayStorage::start()
{
}

void OverlayStorage::finish()
{
}
