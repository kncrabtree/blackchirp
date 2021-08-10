#include "hwsettingsmodel.h"

HWSettingsModel::HWSettingsModel(QString key, QObject *parent) :
    QAbstractItemModel(parent), SettingsStorage(key,Hardware)
{
    d_keys = keys();
    d_arrayKeys = arrayKeys();
    
    //temporary
    discardChanges(true);
    
    QVariantList l{"Key","Value"};
    pu_rootItem = std::make_unique<HWSettingsItem>(l,false,nullptr);
    
    for(auto const &key : d_keys)
    {
        QVariantList data{key,get(key)};
        pu_rootItem->appendChild(new HWSettingsItem(data,true,pu_rootItem.get()));
    }
    
    for(auto const &arrayKey : d_arrayKeys)
    {
        auto v = getArray(arrayKey);
        if(v.size() == 0)
            continue;
        
        QVariantList arrayData{arrayKey};
        auto arrayItem = new HWSettingsItem(arrayData,false,pu_rootItem.get());
        
        for(std::size_t i=0; i<v.size(); ++i)
        {
            auto d = QVariant::fromValue(i);
            QVariantList data{d};
            auto arrayEntry = new HWSettingsItem(data,false,arrayItem);
            
            auto const &m = v.at(i);
            for(auto const &[key,val] : m)
            {
                QVariantList itemData{key,val};
                arrayEntry->appendChild(new HWSettingsItem(itemData,true,arrayEntry));
            }
            
            arrayItem->appendChild(arrayEntry);
        }
        
        pu_rootItem->appendChild(arrayItem);
    }
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
    ///todo
    return false;
}

bool HWSettingsModel::removeRows(int row, int count, const QModelIndex &parent)
{
    ///todo
    return false;
}

Qt::ItemFlags HWSettingsModel::flags(const QModelIndex &index) const
{
    auto item = getItem(index);
    if(item)
    {
        if(item->isEditable())
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
