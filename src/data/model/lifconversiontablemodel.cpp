#include <data/model/lifconversiontablemodel.h>

#include <algorithm>

#include <QComboBox>
#include <QInputDialog>
#include <QMetaEnum>

#include <data/lif/lifconfig.h>
#include <data/storage/enumcsvconvert.h>
#include <data/storage/settingsstorage.h>
#include <hardware/core/liflaser/liffreqconversionstage.h>
#include <hardware/core/liflaser/liflaser.h>
#include <hardware/core/runtimehardwareconfig.h>

using namespace Qt::StringLiterals;
using namespace BC::LifConv;

LifConversionTableModel::LifConversionTableModel(QObject *parent) :
    QAbstractTableModel(parent)
{
    rebuildFromWiring({});
}

void LifConversionTableModel::setFromConfig(const LifConfig &cfg)
{
    // laserKey is provenance-only on the snapshot side (see
    // LifConversionSnapshot); only the wiring/isFinal fields are used here.
    auto snap = LifConversionSnapshot::fromNodes(cfg.conversionNodes(), QString());
    rebuildFromWiring(snap.wiring);
}

void LifConversionTableModel::toConfig(LifConfig &cfg) const
{
    cfg.setConversionNodes(d_nodes, currentLaserKey());
}

void LifConversionTableModel::setFromSnapshot(const LifConversionSnapshot &snap)
{
    rebuildFromWiring(snap.wiring);
}

LifConversionSnapshot LifConversionTableModel::toSnapshot() const
{
    return LifConversionSnapshot::fromNodes(d_nodes, currentLaserKey());
}

LifConversion::AssemblyResult LifConversionTableModel::assemblyResult() const
{
    return LifConversion::assemble(d_nodes);
}

QString LifConversionTableModel::currentLaserKey() const
{
    auto keys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifLaser>();
    return keys.isEmpty() ? QString() : keys.first();
}

BC::LifConv::Op LifConversionTableModel::opAt(int row) const
{
    if(row < 0 || static_cast<std::size_t>(row) >= d_nodes.size())
        return Op::NHG;
    return d_nodes.at(static_cast<std::size_t>(row)).op;
}

QStringList LifConversionTableModel::stageKeysExcluding(const QString &stageKey) const
{
    QStringList out;
    for(const auto &node : d_nodes)
    {
        if(node.stageKey != stageKey)
            out << node.stageKey;
    }
    return out;
}

bool LifConversionTableModel::requestHarmonicChange(const QString &stageKey, int n)
{
    if(n < 1)
        return false;

    auto it = std::find_if(d_nodes.cbegin(), d_nodes.cend(),
                            [&stageKey](const Node &node){ return node.stageKey == stageKey; });
    // Harmonic order is meaningful for NHG only; the gate lives here so every
    // caller (context menu, tests, any future UI) honors it, not just the one
    // widget that disables its menu action.
    if(it == d_nodes.cend() || it->op != Op::NHG)
        return false;

    emit applyHarmonic(stageKey, n);
    return true;
}

void LifConversionTableModel::harmonicApplied(const QString &stageKey)
{
    for(std::size_t i = 0; i < d_nodes.size(); ++i)
    {
        if(d_nodes[i].stageKey != stageKey)
            continue;

        d_nodes[i].n = readHarmonic(stageKey);
        auto idx = index(static_cast<int>(i), HarmonicColumn);
        emit dataChanged(idx, idx);
        emit edited();
        return;
    }
}

void LifConversionTableModel::rebuildFromWiring(const std::vector<BC::LifConv::StageWiring> &wiring)
{
    beginResetModel();

    auto stageKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<LifFreqConversionStage>();

    d_nodes.clear();
    d_nodes.reserve(static_cast<std::size_t>(stageKeys.size()));
    for(const auto &key : stageKeys)
    {
        // op and harmonic live in the same settings group; read both off one
        // snapshot rather than opening the group twice.
        SettingsStorage s(key,SettingsStorage::Hardware);

        Node node;
        node.stageKey = key;
        node.op = BC::CSV::enumFromVariant<Op>(
                    s.get(BC::Key::LifConvStage::op,QVariant::fromValue(Op::NHG)), Op::NHG);
        node.n = s.get(BC::Key::LifConvStage::harmonic,2);

        auto it = std::find_if(wiring.cbegin(), wiring.cend(),
                                [&key](const StageWiring &w){ return w.stageKey == key; });
        if(it != wiring.cend())
        {
            node.inputs = it->inputs;
            node.isFinal = it->isFinal;
        }
        else
        {
            node.inputs = defaultInputs(node.op);
            node.isFinal = false;
        }

        d_nodes.push_back(std::move(node));
    }

    // A wiring entry whose stage is no longer active is dropped; surface it in
    // original wiring order for the preview footer.
    d_droppedStages.clear();
    for(const auto &w : wiring)
    {
        if(!stageKeys.contains(w.stageKey))
            d_droppedStages << w.stageKey;
    }

    endResetModel();
}

std::vector<BC::LifConv::InputRef> LifConversionTableModel::defaultInputs(BC::LifConv::Op op)
{
    InputRef laser;
    laser.type = RefType::Laser;

    if(op == Op::NHG)
        return {laser};

    return {laser,laser};
}

int LifConversionTableModel::readHarmonic(const QString &stageKey)
{
    SettingsStorage s(stageKey,SettingsStorage::Hardware);
    return s.get(BC::Key::LifConvStage::harmonic,2);
}

QVariant LifConversionTableModel::inputEditVariant(const BC::LifConv::InputRef &ref)
{
    return QVariant(QVariantList{static_cast<int>(ref.type), ref.stageKey, ref.fixedCm1});
}

bool LifConversionTableModel::inputRefFromVariant(const QVariant &v, BC::LifConv::InputRef &ref)
{
    auto list = v.toList();
    if(list.size() != 3)
        return false;

    ref.type = static_cast<RefType>(list.at(0).toInt());
    ref.stageKey = list.at(1).toString();
    ref.fixedCm1 = list.at(2).toDouble();
    return true;
}

QString LifConversionTableModel::inputDisplayText(const BC::LifConv::InputRef &ref)
{
    switch(ref.type)
    {
    case RefType::Laser:
        return "Laser"_L1;
    case RefType::Stage:
        return ref.stageKey;
    case RefType::Fixed:
        return u"Fixed: %1 cm⁻¹"_s.arg(ref.fixedCm1,0,'f',3);
    }
    return {};
}

int LifConversionTableModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return static_cast<int>(d_nodes.size());
}

int LifConversionTableModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return NumColumns;
}

QVariant LifConversionTableModel::data(const QModelIndex &index, int role) const
{
    if(!index.isValid() || index.row() < 0 || static_cast<std::size_t>(index.row()) >= d_nodes.size())
        return QVariant();

    const auto &node = d_nodes.at(static_cast<std::size_t>(index.row()));

    if(role == Qt::TextAlignmentRole)
    {
        switch(index.column())
        {
        case StageColumn:
        case Input0Column:
        case Input1Column:
            return QVariant(Qt::AlignLeft|Qt::AlignVCenter);
        default:
            return QVariant(Qt::AlignCenter|Qt::AlignVCenter);
        }
    }

    if(role == Qt::DisplayRole || role == Qt::EditRole)
    {
        switch(index.column())
        {
        case StageColumn:
            return node.stageKey;
        case OpColumn:
            return QString::fromLatin1(QMetaEnum::fromType<Op>().valueToKey(static_cast<int>(node.op)));
        case HarmonicColumn:
            return node.n;
        case Input0Column:
        {
            InputRef ref = node.inputs.size() > 0 ? node.inputs.at(0) : InputRef{};
            return role == Qt::EditRole ? inputEditVariant(ref) : QVariant(inputDisplayText(ref));
        }
        case Input1Column:
        {
            if(node.inputs.size() < 2)
                return role == Qt::EditRole ? QVariant() : QVariant(u"—"_s);
            InputRef ref = node.inputs.at(1);
            return role == Qt::EditRole ? inputEditVariant(ref) : QVariant(inputDisplayText(ref));
        }
        default:
            return QVariant();
        }
    }

    if(role == Qt::CheckStateRole && index.column() == FinalColumn)
        return node.isFinal ? Qt::Checked : Qt::Unchecked;

    if(role == Qt::ToolTipRole)
    {
        switch(index.column())
        {
        case OpColumn:
            return QString("Conversion operation, set by the stage's hardware profile.");
        case HarmonicColumn:
            return QString("Harmonic order (NHG only). Change via the \"Change harmonic…\" context-menu action.");
        case Input0Column:
            return QString("Primary input beam for this stage's conversion.");
        case Input1Column:
            return QString("Secondary input beam; only used by SFG/DFG stages.");
        case FinalColumn:
            return QString("Marks this stage's output as the LIF excitation (output) beam. Exactly one stage must be marked FINAL.");
        default:
            return QVariant();
        }
    }

    return QVariant();
}

bool LifConversionTableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(!index.isValid() || index.row() < 0 || static_cast<std::size_t>(index.row()) >= d_nodes.size())
        return false;

    auto &node = d_nodes[static_cast<std::size_t>(index.row())];

    if(role == Qt::CheckStateRole && index.column() == FinalColumn)
    {
        bool checked = (value.toInt() == Qt::Checked);
        for(auto &n : d_nodes)
            n.isFinal = false;
        node.isFinal = checked;

        emit dataChanged(this->index(0,FinalColumn), this->index(rowCount(QModelIndex())-1,FinalColumn),
                          {Qt::CheckStateRole});
        emit edited();
        return true;
    }

    if(role == Qt::EditRole && (index.column() == Input0Column || index.column() == Input1Column))
    {
        InputRef ref;
        if(!inputRefFromVariant(value, ref))
            return false;

        std::size_t slot = (index.column() == Input0Column) ? 0 : 1;
        if(node.inputs.size() <= slot)
            node.inputs.resize(slot+1);
        node.inputs[slot] = ref;

        emit dataChanged(index,index);
        emit edited();
        return true;
    }

    return false;
}

QVariant LifConversionTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if(orientation == Qt::Horizontal && role == Qt::DisplayRole)
    {
        switch(section)
        {
        case StageColumn:
            return QString("Stage");
        case OpColumn:
            return QString("Op");
        case HarmonicColumn:
            return QString("Harmonic");
        case Input0Column:
            return QString("Input 0");
        case Input1Column:
            return QString("Input 1");
        case FinalColumn:
            return QString("Final");
        }
    }

    return QVariant();
}

Qt::ItemFlags LifConversionTableModel::flags(const QModelIndex &index) const
{
    if(!index.isValid() || index.row() < 0 || static_cast<std::size_t>(index.row()) >= d_nodes.size())
        return Qt::NoItemFlags;

    switch(index.column())
    {
    case StageColumn:
    case OpColumn:
    case HarmonicColumn:
        // Harmonic is gated: read-only in the table, changed only via the
        // "Change harmonic…" context-menu action (requestHarmonicChange()).
        return Qt::ItemIsEnabled;
    case Input0Column:
        return Qt::ItemIsEnabled|Qt::ItemIsEditable;
    case Input1Column:
        if(opAt(index.row()) == Op::NHG)
            return Qt::NoItemFlags;
        return Qt::ItemIsEnabled|Qt::ItemIsEditable;
    case FinalColumn:
        return Qt::ItemIsEnabled|Qt::ItemIsUserCheckable;
    default:
        return Qt::ItemIsEnabled;
    }
}

LifConversionTableDelegate::LifConversionTableDelegate(QObject *parent) : QStyledItemDelegate(parent)
{
}

QWidget *LifConversionTableDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    if(index.column() != LifConversionTableModel::Input0Column &&
       index.column() != LifConversionTableModel::Input1Column)
        return QStyledItemDelegate::createEditor(parent,option,index);

    auto model = dynamic_cast<const LifConversionTableModel*>(index.model());
    if(!model || index.row() < 0 || static_cast<std::size_t>(index.row()) >= model->nodes().size())
        return nullptr;

    const auto stageKey = model->nodes().at(static_cast<std::size_t>(index.row())).stageKey;

    auto cb = new QComboBox(parent);
    cb->addItem("Laser"_L1, QVariant(QVariantList{static_cast<int>(RefType::Laser), QString(), 0.0}));
    for(const auto &key : model->stageKeysExcluding(stageKey))
        cb->addItem(key, QVariant(QVariantList{static_cast<int>(RefType::Stage), key, 0.0}));
    cb->addItem(u"Fixed…"_s, QVariant(QVariantList{static_cast<int>(RefType::Fixed), QString(), 0.0}));
    cb->setEditable(false);

    return cb;
}

void LifConversionTableDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    if(index.column() != LifConversionTableModel::Input0Column &&
       index.column() != LifConversionTableModel::Input1Column)
    {
        QStyledItemDelegate::setEditorData(editor,index);
        return;
    }

    auto cb = dynamic_cast<QComboBox*>(editor);
    if(!cb)
        return;

    auto current = index.model()->data(index,Qt::EditRole).toList();
    if(current.size() != 3)
    {
        cb->setCurrentIndex(0);
        return;
    }

    auto type = static_cast<RefType>(current.at(0).toInt());
    int foundIdx = 0;
    if(type == RefType::Stage)
    {
        auto stageKey = current.at(1).toString();
        for(int i = 0; i < cb->count(); ++i)
        {
            auto data = cb->itemData(i).toList();
            if(data.size() == 3 && static_cast<RefType>(data.at(0).toInt()) == RefType::Stage &&
               data.at(1).toString() == stageKey)
            {
                foundIdx = i;
                break;
            }
        }
    }
    else if(type == RefType::Fixed)
    {
        foundIdx = cb->count()-1;
    }

    cb->setCurrentIndex(foundIdx);
}

void LifConversionTableDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    if(index.column() != LifConversionTableModel::Input0Column &&
       index.column() != LifConversionTableModel::Input1Column)
    {
        QStyledItemDelegate::setModelData(editor,model,index);
        return;
    }

    auto cb = dynamic_cast<QComboBox*>(editor);
    if(!cb)
        return;

    auto data = cb->currentData().toList();
    if(data.size() != 3)
        return;

    auto type = static_cast<RefType>(data.at(0).toInt());
    if(type == RefType::Fixed)
    {
        auto current = model->data(index,Qt::EditRole).toList();
        double seed = (current.size() == 3) ? current.at(2).toDouble() : 0.0;

        bool ok = false;
        double v = QInputDialog::getDouble(cb, "Fixed Input"_L1,
                                            u"Fixed mixing-beam wavenumber (cm⁻¹):"_s,
                                            seed, -1.0e6, 1.0e6, 3, &ok);
        if(!ok)
            return;

        data[2] = v;
    }

    model->setData(index,data);
}

void LifConversionTableDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index)
    editor->setGeometry(option.rect);
}
