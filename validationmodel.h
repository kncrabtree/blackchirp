#ifndef VALIDATIONMODEL_H
#define VALIDATIONMODEL_H

#include <QAbstractTableModel>
#include <QStyledItemDelegate>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QCompleter>

#include <QList>

#include "datastructs.h"

class ValidationModel : public QAbstractTableModel
{
	Q_OBJECT
public:
	ValidationModel(QObject *parent = nullptr);
	~ValidationModel();

    void setFromMap(const QMap<QString, BlackChirp::ValidationItem> l);
	
	// QAbstractItemModel interface
	int rowCount(const QModelIndex &parent) const;
	int columnCount(const QModelIndex &parent) const;
	QVariant data(const QModelIndex &index, int role) const;
	bool setData(const QModelIndex &index, const QVariant &value, int role);
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	bool removeRows(int row, int count, const QModelIndex &parent);
	Qt::ItemFlags flags(const QModelIndex &index) const;
	
    void addNewItem(QString k = QString(""), double min = 0.0, double max = 1.0);
    QList<BlackChirp::ValidationItem> getList();
	
private:
    QList<BlackChirp::ValidationItem> d_validationList;
};

class ValidationDoubleSpinBoxDelegate : public QStyledItemDelegate
{
	Q_OBJECT
public:
    ValidationDoubleSpinBoxDelegate(QObject *parent = nullptr);
	
	// QAbstractItemDelegate interface
	QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	void setEditorData(QWidget *editor, const QModelIndex &index) const;
	void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
	void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

class CompleterLineEditDelegate : public QStyledItemDelegate
{
	Q_OBJECT
public:
	CompleterLineEditDelegate(QObject *parent = nullptr);
	
	// QAbstractItemDelegate interface
	QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	void setEditorData(QWidget *editor, const QModelIndex &index) const;
	void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
	void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

#endif // VALIDATIONMODEL_H
