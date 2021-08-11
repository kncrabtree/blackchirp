#include "hwsettingsmodel.h"

#include <hardware/core/hardwareobject.h>

HWSettingsModel::HWSettingsModel(QString key, QStringList forbiddenKeys, QObject *parent) :
    QAbstractItemModel(parent), SettingsStorage(key,Hardware)
{
    auto k = keys();
    auto ak = arrayKeys();

    QVector<QVariant> l{"Key","Value"};
    pu_rootItem = std::make_unique<HWSettingsItem>(l,false,false,nullptr);

    forbiddenKeys.append({
        BC::Key::HW::commType, BC::Key::HW::connected,BC::Key::HW::key,
        BC::Key::HW::name,BC::Key::HW::threaded,BC::Key::Custom::comm
    });
    
    for(auto const &key : k)
    {
        if(forbiddenKeys.contains(key))
            continue;

        QVector<QVariant> data{key,get(key)};
        pu_rootItem->appendChild(new HWSettingsItem(data,true,false,pu_rootItem.get()));
    }
    
    for(auto const &arrayKey : ak)
    {
        if(forbiddenKeys.contains(arrayKey))
            continue;

        auto v = getArray(arrayKey);
        if(v.size() == 0)
            continue;
        
        QVector<QVariant> arrayData{arrayKey,""};
        auto arrayItem = new HWSettingsItem(arrayData,false,false,pu_rootItem.get());
        
        for(std::size_t i=0; i<v.size(); ++i)
        {
            auto d = QVariant::fromValue(i);
            QVector<QVariant> data{d,""};
            auto arrayEntry = new HWSettingsItem(data,false,true,arrayItem);
            
            auto const &m = v.at(i);
            for(auto const &[key,val] : m)
            {
                QVector<QVariant> itemData{key,val};
                arrayEntry->appendChild(new HWSettingsItem(itemData,true,false,arrayEntry));
            }
            
            arrayItem->appendChild(arrayEntry);
        }
        
        pu_rootItem->appendChild(arrayItem);
    }
}

void HWSettingsModel::saveChanges()
{
    for(int i=0; i<pu_rootItem->childCount(); ++i)
    {
        auto item = pu_rootItem->childAt(i);
        if(!item)
            continue;

        if(item->childCount() == 0)
            set(item->data(0).toString(),item->data(1));
        else
        {
            auto arrayKey = item->data(0).toString();
            setArray(arrayKey,{});
            std::vector<SettingsMap> vec;
            for(int j=0; j<item->childCount(); ++j)
            {
                auto arrayItem = item->childAt(j);
                if(!arrayItem)
                    continue;

                SettingsMap m;
                for(int k=0; k<arrayItem->childCount(); ++k)
                    m.insert_or_assign(arrayItem->childAt(k)->data(0).toString(),
                                       arrayItem->childAt(k)->data(1));

                vec.push_back(m);
            }
            setArray(arrayKey,vec);
        }
    }

    save();
}


QModelIndex HWSettingsModel::index(int row, int column, const QModelIndex &parent) const
{
    if(!hasIndex(row,column,parent))
        return QModelIndex();
    
    HWSettingsItem *parentItem;
    if(parent.isValid())
        parentItem = static_cast<HWSettingsItem*>(parent.internalPointer());
    else
        parentItem = pu_rootItem.get();
    
    auto childItem = parentItem->childAt(row);
    
    return childItem ? createIndex(row,column,childItem) : QModelIndex();
}

QModelIndex HWSettingsModel::parent(const QModelIndex &child) const
{
    if(!child.isValid())
        return QModelIndex();
    
    auto childItem = static_cast<HWSettingsItem*>(child.internalPointer());
    auto parentItem = childItem->parent();
    
    if(parentItem == pu_rootItem.get())
        return QModelIndex();
    
    return createIndex(parentItem->row(),0,parentItem);
}

int HWSettingsModel::rowCount(const QModelIndex &parent) const
{
    return getItem(parent)->childCount();
}

int HWSettingsModel::columnCount(const QModelIndex &parent) const
{
    return getItem(parent)->columnCount();
}

bool HWSettingsModel::hasChildren(const QModelIndex &parent) const
{
    return getItem(parent)->childCount() > 0;
}

QVariant HWSettingsModel::data(const QModelIndex &index, int role) const
{
    QVariant out;
    
    if(!index.isValid())
        return out;
    
    if(role == Qt::DisplayRole || role == Qt::EditRole)
        return getItem(index)->data(index.column());
    
    return out;
}

bool HWSettingsModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(!index.isValid())
        return false;
    
    if(role == Qt::EditRole && index.column() == 1)
    {
        auto item = getItem(index);
        if(item)
        {
            bool out = item->setData(1,value);
            if(out)
                emit dataChanged(index,index,{role});
            
            return out;
        }
    }
    
    return false;
}

QVariant HWSettingsModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return pu_rootItem->data(section);

    return QVariant();
}

bool HWSettingsModel::insertRows(int row, int count, const QModelIndex &parent)
{
    //Rows can only be inserted/removed for array values, and we need to ensure that new items that are created
    //have the correct keys available for the user to edit.
    //We get the keys from the first child array item (since they should be the same for all items).
    if(!parent.isValid())
        return false;

    auto parentItem = getItem(parent);
    if(!parentItem)
        return false;
    if(parentItem->childCount() == 0)
        return false;

    //get the list of keys for the array value in settings
    auto keys = parentItem->childAt(0)->childKeys();

    beginInsertRows(parent,row,row+count-1);
    bool success = true;
    for(int i=0; i<count; ++i)
        success &= parentItem->addChild(row,keys);
    endInsertRows();

    emit dataChanged(index(0,0,parent),index(rowCount(parent),0,parent),{Qt::DisplayRole});

    return success;
}

bool HWSettingsModel::removeRows(int row, int count, const QModelIndex &parent)
{
    if(!parent.isValid())
        return false;

    auto parentItem = getItem(parent);
    if(!parentItem)
        return false;

    auto childItem = parentItem->childAt(row);
    if(!childItem || !childItem->canAddChildren())
        return false;

    bool success = true;
    beginRemoveRows(parent,row,row+count-1);
    for(int i=0;i<count;++i)
        success &= parentItem->removeChild(row);
    endRemoveRows();

    emit dataChanged(index(0,0,parent),index(rowCount(parent),0,parent),{Qt::DisplayRole});

    return success;
}

Qt::ItemFlags HWSettingsModel::flags(const QModelIndex &index) const
{
    auto item = getItem(index);
    if(item)
    {
        if(item->isEditable() && index.column() == 1)
            return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
        else
            return QAbstractItemModel::flags(index);
    }
    
    return 0;
}

HWSettingsItem *HWSettingsModel::getItem(const QModelIndex &index) const
{
    if(index.isValid())
    {
        auto item = static_cast<HWSettingsItem*>(index.internalPointer());
        if(item)
            return item;
    }
    
    return pu_rootItem.get();
}
