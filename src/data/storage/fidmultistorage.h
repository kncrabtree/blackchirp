#ifndef FIDMULTISTORAGE_H
#define FIDMULTISTORAGE_H

#include <data/storage/fidstoragebase.h>

class QMutex;

class FidMultiStorage : public FidStorageBase
{
public:
    FidMultiStorage(int numRecords, int num, QString path);

    void setNumSegments(int s) { d_numSegments = s; }
    int numSegments() const { return d_numSegments; }

    // FidStorageBase interface
    int getCurrentIndex() override;

protected:
    void _advance() override;

private:
    int d_currentSegment{0};
    int d_numSegments{0};
};

#endif // FIDMULTISTORAGE_H
