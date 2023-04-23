#include <data/model/validationmodel.h>

#include <QDoubleValidator>


ValidationModel::ValidationModel(QObject *parent) :
     QAbstractTableModel(parent), SettingsStorage(BC::Key::Validation::key)
{
    using namespace BC::Key::Validation;
    auto n = getArraySize(items);
    for(std::size_t i=0; i<n; ++i)
    {
        auto ok = getArrayValue(items,i,objKey,QString(""));
        auto vk = getArrayValue(items,i,valKey,QString(""));
        auto minv = getArrayValue(items,i,min,0.0);
        auto maxv = getArrayValue(items,i,max,1.0);

        if(!ok.isEmpty() && !vk.isEmpty())
            d_modelData.append({ok,vk,minv,maxv});
    }
}

ValidationModel::~ValidationModel()
{
    using namespace BC::Key::Validation;
    setArray(items,{},false);
    for(int i=0; i<d_modelData.size(); ++i)
    {
        appendArrayMap(items,{
                           {objKey,d_modelData.at(i).at(0)},
                           {valKey,d_modelData.at(i).at(1)},
                           {min,d_modelData.at(i).at(2)},
                           {max,d_modelData.at(i).at(3)}
                       },false);
    }
}


int ValidationModel::rowCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)
	
    return d_modelData.size();
}

int ValidationModel::columnCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)
	
    return 4;
}

QVariant ValidationModel::data(const QModelIndex &index, int role) const
{
	QVariant out;
    if(index.row() < 0 || index.row() >= d_modelData.size())
		return out;

    if(role == Qt::DisplayRole)
        out = d_modelData.at(index.row()).at(index.column());
    if(role == Qt::EditRole)
	{
		switch (index.column()) {
        case 0:
        case 1:
            out = d_modelData.at(index.row()).at(index.column()).toString();
            break;
        case 2:
        case 3:
            out = d_modelData.at(index.row()).at(index.column()).toDouble();
            break;
        default:
			break;
        }
	}
	
	return out;
}

bool ValidationModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	if(role != Qt::EditRole)
		return false;
	
    if(index.row() < 0 || index.row() >= d_modelData.size())
		return false;

    d_modelData[index.row()][index.column()] = value;

    emit dataChanged(index,index);
	
    return true;
}

QVariant ValidationModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	QVariant out;
	
	if(orientation == Qt::Horizontal)
	{
		if(role == Qt::DisplayRole)
		{
			switch (section) {
			case 0:
                out = QString("Object Key");
				break;
            case 1:
                out = QString("Value Key");
                break;
            case 2:
				out = QString("Min");
				break;
            case 3:
				out = QString("Max");
				break;
			default:
				break;
			}
		}
		else if(role == Qt::ToolTipRole)
		{
			switch (section) {
			case 0:
                out = QString("The object key for the validation item.\nA typical data name is ObjKey.subkey.[name].ValueKey (where name is optional).");
				break;
            case 1:
                out = QString("The value key for the validation item.\nA typical data name is ObjKey.subkey.[name].ValueKey (where name is optional).");
                break;
            case 2:
                out = QString("Minimum valid value. If the value read is lower, the experiment will be aborted.");
				break;
            case 3:
                out = QString("Maximum valid value. If the value read is higher, the experiment will be aborted.");
				break;
			default:
				break;
			}
		}
	}
	else if(orientation == Qt::Vertical)
	{
		if(role == Qt::DisplayRole)
			out = section+1;
	}
	
	return out;
}

bool ValidationModel::removeRows(int row, int count, const QModelIndex &parent)
{
    if(row < 0 || row+count > d_modelData.size() || d_modelData.isEmpty())
	        return false;
	
	beginRemoveRows(parent,row,row+count-1);
	for(int i=0; i<count; i++)
        d_modelData.removeAt(row);
	endRemoveRows();
	
	return true;
}

Qt::ItemFlags ValidationModel::flags(const QModelIndex &index) const
{
    if(index.row() < 0 || index.row() >= d_modelData.size())
		return Qt::NoItemFlags;
	
    return Qt::ItemIsEnabled | Qt::ItemIsEditable | Qt::ItemIsSelectable;
}

void ValidationModel::addNewItem()
{
    beginInsertRows(QModelIndex(),d_modelData.size(),d_modelData.size());
    d_modelData.append({"","",0.0,1.0});
    endInsertRows();
}

ValidationDelegate::ValidationDelegate(QObject *parent) :
     QStyledItemDelegate(parent)
{
	
}

QWidget *ValidationDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
	Q_UNUSED(option)

    if(index.column() == 2 || index.column() == 3)
    {
        auto out = new QLineEdit(parent);
        QValidator *v = new QDoubleValidator(out);
        out->setValidator(v);
        return out;
    }

    auto &m = static_cast<const ValidationModel*>(index.model())->d_validationKeys;

    QStringList keys;
    if(index.column() == 0)
    {
        for(auto it = m.cbegin(); it != m.cend(); ++it)
        {
            if(!it->first.isEmpty())
                keys.append(it->first);
        }
    }
    else
    {
        auto k1 = index.model()->data(index.model()->index(index.row(),0)).toString();
        auto it = m.find(k1);
        if(it != m.end())
            keys.append(it->second);
    }
	
    auto out = new QComboBox(parent);
	
    if(!keys.isEmpty())
	{
        QCompleter *comp = new QCompleter(keys,out);
		comp->setCaseSensitivity(Qt::CaseInsensitive);
        comp->setCompletionMode(QCompleter::UnfilteredPopupCompletion);
        out->setCompleter(comp);

        for(auto k : keys)
            out->addItem(k);
	}

    return out;
}

void ValidationDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    if(index.column() == 0 || index.column() == 1)
        static_cast<QComboBox*>(editor)->setCurrentIndex(
                static_cast<QComboBox*>(editor)->findText(
                    index.model()->data(index,Qt::EditRole).toString()));
    else
        static_cast<QLineEdit*>(editor)->setText(index.model()->data(index,Qt::EditRole).toString());

}

void ValidationDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    if(index.column() == 0 || index.column() == 1)
        model->setData(index,static_cast<QComboBox*>(editor)->currentText());
    else
        model->setData(index,static_cast<QLineEdit*>(editor)->text().toDouble());
}

void ValidationDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
	editor->setGeometry(option.rect);
	Q_UNUSED(index)
}
