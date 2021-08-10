#ifndef HWSETTINGSMODEL_H
#define HWSETTINGSMODEL_H

#include <QAbstractItemModel>
#include <memory>
#include <data/storage/settingsstorage.h>

class HWSettingsItem;

class HWSettingsModel : public QAbstractItemModel, public SettingsStorage
{
    Q_OBJECT
public:
    HWSettingsModel(QString key, QObject *parent = nullptr);
    
private:
    QStringList d_keys;
    QStringList d_arrayKeys;
    std::unique_ptr<HWSettingsItem> pu_rootItem;
    
    // QAbstractItemModel interface
public:
    QModelIndex index(int row, int column, const QModelIndex &parent) const override;
    QModelIndex parent(const QModelIndex &child) const override;
    int rowCount(const QModelIndex &parent) const override;
    int columnCount(const QModelIndex &parent) const override;
    bool hasChildren(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    bool insertRows(int row, int count, const QModelIndex &parent) override;
    bool removeRows(int row, int count, const QModelIndex &parent) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    
private:
    HWSettingsItem *getItem(const QModelIndex &index) const;
};

class HWSettingsItem
{
public:
    HWSettingsItem(QVariantList data, bool editable = true, HWSettingsItem *parent = nullptr) :
        p_parent(parent), d_data(data), d_isEditable(editable) {}
    ~HWSettingsItem() { qDeleteAll(d_children); }
    
    void appendChild(HWSettingsItem *item) { d_children.append(item); }
    
    HWSettingsItem *parent() const { return p_parent; }
    HWSettingsItem *childAt(int row) const { return d_children.value(row); }
    int childCount() const { return d_children.size(); }
    int columnCount() const { return d_data.size(); }
    QVariant data(int column) const { return d_data.value(column); }
    bool setData(int column, QVariant data) { 
        if(column != 1 || column >= d_data.size())
            return false;
         d_data[column] = data;
         return true;
    }
    int row() const {
        return p_parent ? p_parent->d_children.indexOf(const_cast<HWSettingsItem*>(this)) : 0;
    }
    bool isEditable() const { return d_isEditable; }
      
private:
    HWSettingsItem *p_parent;
    QVector<HWSettingsItem*> d_children;
    QVariantList d_data;
    bool d_isEditable;
    
};
    
    

#endif // HWSETTINGSMODEL_H
