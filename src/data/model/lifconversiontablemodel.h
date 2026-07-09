#ifndef LIFCONVERSIONTABLEMODEL_H
#define LIFCONVERSIONTABLEMODEL_H

#include <QAbstractTableModel>
#include <QStyledItemDelegate>

#include <vector>

#include <data/lif/lifconversion.h>
#include <data/loadout/lifconversionsnapshot.h>

class LifConfig;

/*!
 * \brief Table model for the per-experiment LIF frequency-conversion DAG.
 *
 * Rows are the active \c LifFreqConversionStage hardware keys (from
 * \c RuntimeHardwareConfig). Each row's \c op/harmonic order are read from
 * that stage's \c SettingsStorage hardware snapshot — the model never talks
 * to a live threaded device — while the input wiring and FINAL marker are
 * per-experiment state edited directly in the table. The joined
 * \c BC::LifConv::Node list (\c d_nodes) is the model's sole state; it is
 * assembled locally via \c LifConversion::assemble() on demand for the
 * widget's preview footer, and bridged to/from \c LifConfig and
 * \c LifConversionSnapshot (preset wiring) via setFromConfig()/toConfig()
 * and setFromSnapshot()/toSnapshot().
 *
 * A harmonic-order change is gated: the Harmonic column is read-only in the
 * table proper (flags()), and can only be requested via
 * requestHarmonicChange(), which emits applyHarmonic() rather than editing
 * the model directly. The model's state is only updated once a later stage
 * (HardwareManager, over a channel wired up elsewhere) confirms success by
 * calling harmonicApplied().
 */
class LifConversionTableModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    explicit LifConversionTableModel(QObject *parent = nullptr);

    enum Column {
        StageColumn,
        OpColumn,
        HarmonicColumn,
        Input0Column,
        Input1Column,
        FinalColumn,
        NumColumns
    };

    //! Seed the joined node list from an experiment's already-joined conversion nodes.
    void setFromConfig(const LifConfig &cfg);
    //! Write the joined node list into \a cfg (op/n from hardware, wiring from the table).
    void toConfig(LifConfig &cfg) const;

    //! Seed wiring (inputs/FINAL only) from a preset snapshot; op/n are always re-read live.
    void setFromSnapshot(const LifConversionSnapshot &snap);
    //! Build a wiring-only snapshot from the current table state.
    LifConversionSnapshot toSnapshot() const;

    //! The model's joined node list (op/n from hardware, inputs/isFinal from the table).
    const std::vector<BC::LifConv::Node> &nodes() const { return d_nodes; }

    //! Assemble the current node list; \c ok is false with errorString set when the graph is invalid.
    LifConversion::AssemblyResult assemblyResult() const;

    //! Wired stages that cannot be reproduced on the current hardware (hwKey no longer active, or op changed the required input arity); non-empty means the loaded preset was rejected and the configuration cleared (surfaced by the preview footer).
    QStringList incompatibleStages() const { return d_incompatibleStages; }

    //! Active LifLaser hwKey (used for the wiring's Laser-typed inputs and preset provenance).
    QString currentLaserKey() const;

    //! Conversion operation of the stage at \a row.
    BC::LifConv::Op opAt(int row) const;

    //! Active stage keys other than \a stageKey, for populating an input combo.
    QStringList stageKeysExcluding(const QString &stageKey) const;

    /*!
     * \brief Request a gated harmonic-order change for \a stageKey.
     *
     * Emits applyHarmonic() for a later stage to route to
     * \c LifFreqConversionStage::setHarmonicOrder() via \c HardwareManager;
     * does **not** modify the model. Returns \c false without emitting when
     * \a stageKey does not name an active row or \a n is not a valid
     * harmonic order (< 1).
     */
    bool requestHarmonicChange(const QString &stageKey, int n);

public slots:
    /*!
     * \brief Re-read \a stageKey's harmonic order from its settings snapshot
     *        and re-join it into the model, firing edited().
     *
     * Called by a later stage once a requestHarmonicChange() request has
     * been confirmed successful on hardware; a no-op if \a stageKey does
     * not name an active row.
     */
    void harmonicApplied(const QString &stageKey);

signals:
    //! Fired whenever the joined node list changes via a table edit or harmonicApplied().
    void edited();
    //! A gated harmonic-order change was requested for \a stageKey; never auto-applied.
    void applyHarmonic(QString stageKey, int n);

private:
    std::vector<BC::LifConv::Node> d_nodes;
    QStringList d_incompatibleStages;

    //! Rebuild d_nodes from the currently active stages, overlaying \a wiring where present.
    void rebuildFromWiring(const std::vector<BC::LifConv::StageWiring> &wiring);

    static std::vector<BC::LifConv::InputRef> defaultInputs(BC::LifConv::Op op);
    static int readHarmonic(const QString &stageKey);

    //! Encode an InputRef as the {type,stageKey,fixedCm1} EditRole variant, and its inverse.
    static QVariant inputEditVariant(const BC::LifConv::InputRef &ref);
    static bool inputRefFromVariant(const QVariant &v, BC::LifConv::InputRef &ref);
    static QString inputDisplayText(const BC::LifConv::InputRef &ref);

    // QAbstractItemModel interface
public:
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
};

/*!
 * \brief Delegate for LifConversionTableModel's Input 0/Input 1 columns.
 *
 * Presents a combo box of "Laser" / other active stages / "Fixed…" (which
 * prompts for the fixed cm⁻¹ value via QInputDialog on selection). All other
 * columns fall back to the base QStyledItemDelegate; the Harmonic column is
 * never opened for inline editing (see LifConversionTableModel::flags()).
 */
class LifConversionTableDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit LifConversionTableDelegate(QObject *parent = nullptr);

    // QAbstractItemDelegate interface
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const override;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
};

#endif // LIFCONVERSIONTABLEMODEL_H
