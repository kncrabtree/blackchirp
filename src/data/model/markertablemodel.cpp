#include "markertablemodel.h"

#include <hardware/optional/chirpsource/awg.h>
#include <hardware/core/runtimehardwareconfig.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLineEdit>

// ============================================================================
// MarkerTableModel
// ============================================================================

MarkerTableModel::MarkerTableModel(QObject *parent)
    : QAbstractTableModel(parent), SettingsStorage(BC::Key::MarkerTableModel::key),
      d_markerCount(0)
{
    auto awgKeys = RuntimeHardwareConfig::constInstance().getActiveKeys<AWG>();
    if(!awgKeys.isEmpty())
    {
        SettingsStorage s(awgKeys.first(), SettingsStorage::Hardware);
        d_markerCount = s.get(BC::Key::AWG::markerCount, 0);
    }

    // Read last-used marker configuration from settings
    using namespace BC::Key::MarkerTableModel;
    auto num = static_cast<int>(getArraySize(markers));
    for(int i = 0; i < qMin(num, d_markerCount); ++i)
    {
        MarkerChannel m;
        m.name     = getArrayValue(markers, i, chName, QString("Marker %1").arg(i));
        m.role     = static_cast<MarkerRole>(getArrayValue(markers, i, chRole,
                                             static_cast<int>(MarkerRole::Custom)));
        m.startTime = getArrayValue(markers, i, chStart, -0.5);
        m.endTime   = getArrayValue(markers, i, chEnd,    0.5);
        m.enabled   = getArrayValue(markers, i, chEnabled, true);
        m.timingMode = MarkerChannel::ChirpRelative;
        d_channels.append(m);
    }

    // Fill any channels not yet in settings with defaults
    for(int i = d_channels.size(); i < d_markerCount; ++i)
    {
        MarkerChannel m;
        m.timingMode = MarkerChannel::ChirpRelative;
        m.startTime  = -0.5;
        m.endTime    = 0.5;
        m.enabled    = true;
        if(i == 0)      { m.name = QStringLiteral("Protection"); m.role = MarkerRole::Protection; }
        else if(i == 1) { m.name = QStringLiteral("Gate");       m.role = MarkerRole::Gate;       }
        else            { m.name = QString("Marker %1").arg(i);  m.role = MarkerRole::Custom;     }
        d_channels.append(m);
    }
}

MarkerTableModel::~MarkerTableModel()
{
    saveToSettings();
}

void MarkerTableModel::saveToSettings()
{
    using namespace BC::Key::MarkerTableModel;
    setArray(markers, {});
    for(const auto &m : d_channels)
    {
        appendArrayMap(markers, {
            {chName,    m.name},
            {chRole,    static_cast<int>(m.role)},
            {chStart,   m.startTime},
            {chEnd,     m.endTime},
            {chEnabled, m.enabled}
        });
    }
}

int MarkerTableModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return d_channels.size();
}

int MarkerTableModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 6; // Ch, Name, Role, Start (μs), End (μs), Enabled
}

QVariant MarkerTableModel::data(const QModelIndex &index, int role) const
{
    if(!index.isValid() || index.row() >= d_channels.size())
        return QVariant();

    const auto &m = d_channels.at(index.row());

    if(role == Qt::TextAlignmentRole)
        return Qt::AlignCenter;

    auto roleString = [](MarkerRole r) -> QString {
        switch(r) {
        case MarkerRole::Protection: return QStringLiteral("Protection");
        case MarkerRole::Gate:       return QStringLiteral("Gate");
        case MarkerRole::Trigger:    return QStringLiteral("Trigger");
        default:                     return QStringLiteral("Custom");
        }
    };

    if(role == Qt::DisplayRole)
    {
        switch(index.column()) {
        case 0: return index.row();
        case 1: return m.name;
        case 2: return roleString(m.role);
        case 3: return QString::number(m.startTime, 'f', 3);
        case 4: return QString::number(m.endTime, 'f', 3);
        case 5: return m.enabled ? QStringLiteral("Yes") : QStringLiteral("No");
        default: return QVariant();
        }
    }

    if(role == Qt::EditRole)
    {
        switch(index.column()) {
        case 1: return m.name;
        case 2: return static_cast<int>(m.role);
        case 3: return m.startTime;
        case 4: return m.endTime;
        case 5: return m.enabled;
        default: return QVariant();
        }
    }

    return QVariant();
}

QVariant MarkerTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if(role == Qt::DisplayRole)
    {
        if(orientation == Qt::Vertical)
            return QVariant();

        switch(section) {
        case 0: return QStringLiteral("Ch");
        case 1: return QStringLiteral("Name");
        case 2: return QStringLiteral("Role");
        case 3: return QString::fromUtf8("Start (μs)");
        case 4: return QString::fromUtf8("End (μs)");
        case 5: return QStringLiteral("Enabled");
        default: return QVariant();
        }
    }

    if(role == Qt::ToolTipRole && orientation == Qt::Horizontal)
    {
        switch(section) {
        case 0: return QStringLiteral("Marker channel index (0-based)");
        case 1: return QStringLiteral("User-defined label for this marker channel");
        case 2: return QStringLiteral("Marker role: Protection, Gate, Trigger, or Custom");
        case 3: return QStringLiteral("Start time relative to chirp start (negative = before chirp)");
        case 4: return QStringLiteral("End time relative to chirp end (positive = after chirp)");
        case 5: return QStringLiteral("Whether this marker channel is active");
        default: return QVariant();
        }
    }

    return QVariant();
}

bool MarkerTableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(role != Qt::EditRole)
        return false;

    if(!index.isValid() || index.row() >= d_channels.size())
        return false;

    auto &m = d_channels[index.row()];
    switch(index.column()) {
    case 1: m.name = value.toString(); break;
    case 2: m.role = static_cast<MarkerRole>(value.toInt()); break;
    case 3: m.startTime = value.toDouble(); break;
    case 4: m.endTime = value.toDouble(); break;
    case 5: m.enabled = value.toBool(); break;
    default: return false;
    }

    emit dataChanged(index, index);
    emit modelChanged();
    return true;
}

Qt::ItemFlags MarkerTableModel::flags(const QModelIndex &index) const
{
    if(!index.isValid() || index.row() >= d_channels.size())
        return Qt::NoItemFlags;

    // Channel column is display-only
    if(index.column() == 0)
        return Qt::ItemIsEnabled | Qt::ItemIsSelectable;

    return Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsEditable;
}

void MarkerTableModel::setFromChirpConfig(const ChirpConfig &cc)
{
    const auto &existing = cc.markerChannels();

    // If the config carries marker data (re-running a previous experiment),
    // replace d_channels with those values and persist them as the new last-used.
    // If the config is empty (new experiment), keep d_channels as-is —
    // the constructor already populated it from settings (last-used) or defaults.
    if(existing.isEmpty())
        return;

    beginResetModel();
    d_channels.clear();
    for(int i = 0; i < d_markerCount; ++i)
    {
        if(i < existing.size())
            d_channels.append(existing.at(i));
        else
        {
            // The experiment had fewer channels than the current AWG supports;
            // fill remaining with settings-backed values if available, else defaults.
            MarkerChannel m;
            m.timingMode = MarkerChannel::ChirpRelative;
            m.startTime  = -0.5;
            m.endTime    = 0.5;
            m.enabled    = true;
            if(i == 0)      { m.name = QStringLiteral("Protection"); m.role = MarkerRole::Protection; }
            else if(i == 1) { m.name = QStringLiteral("Gate");       m.role = MarkerRole::Gate;       }
            else            { m.name = QString("Marker %1").arg(i);  m.role = MarkerRole::Custom;     }
            d_channels.append(m);
        }
    }
    endResetModel();
    emit modelChanged();
}

// ============================================================================
// MarkerDelegate
// ============================================================================

MarkerDelegate::MarkerDelegate(QObject *parent) : QStyledItemDelegate(parent)
{
}

QWidget *MarkerDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &,
                                      const QModelIndex &index) const
{
    switch(index.column())
    {
    case 1:
    {
        auto *ed = new QLineEdit(parent);
        return ed;
    }
    case 2:
    {
        auto *cb = new QComboBox(parent);
        cb->addItem(QStringLiteral("Custom"),    static_cast<int>(MarkerRole::Custom));
        cb->addItem(QStringLiteral("Protection"),static_cast<int>(MarkerRole::Protection));
        cb->addItem(QStringLiteral("Gate"),      static_cast<int>(MarkerRole::Gate));
        cb->addItem(QStringLiteral("Trigger"),   static_cast<int>(MarkerRole::Trigger));
        return cb;
    }
    case 3:
    case 4:
    {
        auto *sb = new QDoubleSpinBox(parent);
        sb->setRange(-100.0, 100.0);
        sb->setDecimals(3);
        sb->setSingleStep(0.01);
        sb->setSuffix(QString::fromUtf8(" μs"));
        return sb;
    }
    case 5:
    {
        auto *cb = new QCheckBox(parent);
        return cb;
    }
    default:
        return nullptr;
    }
}

void MarkerDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    auto val = index.model()->data(index, Qt::EditRole);
    switch(index.column())
    {
    case 1:
        static_cast<QLineEdit*>(editor)->setText(val.toString());
        break;
    case 2:
    {
        auto *cb = static_cast<QComboBox*>(editor);
        int role = val.toInt();
        for(int i = 0; i < cb->count(); ++i)
        {
            if(cb->itemData(i).toInt() == role)
            {
                cb->setCurrentIndex(i);
                break;
            }
        }
        break;
    }
    case 3:
    case 4:
        static_cast<QDoubleSpinBox*>(editor)->setValue(val.toDouble());
        break;
    case 5:
        static_cast<QCheckBox*>(editor)->setChecked(val.toBool());
        break;
    default:
        break;
    }
}

void MarkerDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                  const QModelIndex &index) const
{
    switch(index.column())
    {
    case 1:
        model->setData(index, static_cast<QLineEdit*>(editor)->text(), Qt::EditRole);
        break;
    case 2:
    {
        auto *cb = static_cast<QComboBox*>(editor);
        model->setData(index, cb->currentData(), Qt::EditRole);
        break;
    }
    case 3:
    case 4:
    {
        auto *sb = static_cast<QDoubleSpinBox*>(editor);
        sb->interpretText();
        model->setData(index, sb->value(), Qt::EditRole);
        break;
    }
    case 5:
        model->setData(index, static_cast<QCheckBox*>(editor)->isChecked(), Qt::EditRole);
        break;
    default:
        break;
    }
}

void MarkerDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option,
                                          const QModelIndex &) const
{
    editor->setGeometry(option.rect);
}
