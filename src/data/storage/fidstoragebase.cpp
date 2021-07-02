#include "fidstoragebase.h"

FidStorageBase::FidStorageBase(const QString path, int numRecords) :
    d_path(path), d_numRecords(numRecords)
{
}

FidStorageBase::~FidStorageBase()
{
}

void FidStorageBase::advance()
{
    save();
    _advance();
}

bool FidStorageBase::save()
{
    //if path isn't set, then data can't be saved
    //Don't throw an error; this is probably intentional (peak up mode)
    if(d_path.isEmpty())
        return true;

    auto l = getCurrentFidList();
    auto i = getCurrentIndex();
    l.detach();
    for(int i=0; i<l.size(); ++i)
        l[i].detach();

    return saveFidList(l,i);
}

bool FidStorageBase::saveFidList(const FidList l, int i)
{
    ///todo
    (void)l;
    (void)i;
#pragma message("Write implememtnation for saveFidList")
    return true;
}

FidList FidStorageBase::newFidList() const
{
    FidList out;
    out.resize(d_numRecords);
    return out;
}

