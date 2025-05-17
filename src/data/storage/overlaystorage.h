#ifndef OVERLAYSTORAGE_H
#define OVERLAYSTORAGE_H

#include "datastoragebase.h"

class OverlayStorage : public DataStorageBase
{
public:
    OverlayStorage(int number, QString path);
    
    // DataStorageBase interface
public:
    void advance() override;
    void save() override;
    void start() override;
    void finish() override;
};

#endif // OVERLAYSTORAGE_H
