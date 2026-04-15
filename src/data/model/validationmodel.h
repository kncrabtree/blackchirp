#ifndef VALIDATIONMODEL_H
#define VALIDATIONMODEL_H

#include <QAbstractTableModel>
#include <QStyledItemDelegate>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QCompleter>

#include <data/experiment/experimentvalidator.h>
#include <data/storage/settingsstorage.h>

namespace BC::Key::Validation {
inline constexpr QLatin1StringView key{"ValidationTable"};
inline constexpr QLatin1StringView items{"items"};
inline constexpr QLatin1StringView objKey{"objKey"};
inline constexpr QLatin1StringView valKey{"valueKey"};
inline constexpr QLatin1StringView min{"min"};
inline constexpr QLatin1StringView max{"max"};
}

class ValidationModel : public QAbstractTableModel, public SettingsStorage
{
	Q_OBJECT
public:
	ValidationModel(QObject *parent = nullptr);
	~ValidationModel();
	
	// QAbstractItemModel interface
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;
	QVariant data(const QModelIndex &index, int role) const;
	bool setData(const QModelIndex &index, const QVariant &value, int role);
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	bool removeRows(int row, int count, const QModelIndex &parent);
	Qt::ItemFlags flags(const QModelIndex &index) const;
	
    void addNewItem();
    std::map<QString,QStringList> d_validationKeys;
	
private:
    QVector<QVariantList> d_modelData;
};

class ValidationDelegate : public QStyledItemDelegate
{
	Q_OBJECT
public:
    ValidationDelegate(QObject *parent = nullptr);
	
	// QAbstractItemDelegate interface
	QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	void setEditorData(QWidget *editor, const QModelIndex &index) const;
	void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
	void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

#endif // VALIDATIONMODEL_H
