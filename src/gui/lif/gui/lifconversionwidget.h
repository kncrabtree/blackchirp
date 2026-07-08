#ifndef LIFCONVERSIONWIDGET_H
#define LIFCONVERSIONWIDGET_H

#include <QIcon>
#include <QWidget>

#include <data/loadout/hardwareloadout.h>
#include <data/model/lifconversiontablemodel.h>

class QComboBox;
class QLabel;
class QPushButton;
class QTableView;
class LifConfig;

namespace BC::Key::LifConversionWidget {
inline constexpr QLatin1StringView key{"LifConversionWidget"};
}

/*!
 * \brief Table + preset bar + preview footer for the per-experiment LIF
 *        frequency-conversion topology.
 *
 * Hosts a LifConversionTableModel/LifConversionTableDelegate table view, a
 * preset bar wired to the LoadoutManager LIF preset API (mirrors
 * FtmwConfigWidget's preset bar), and a live preview footer that renders the
 * assembled chain expression, output range, any assembly error, and any
 * stages dropped during seeding (drift between a loaded config/preset and
 * the currently active hardware).
 *
 * The widget itself never talks to hardware: edited() and applyHarmonic()
 * are simply re-exposed from the model for a later page host to connect to
 * the experiment config and to HardwareManager, respectively.
 */
class LifConversionWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LifConversionWidget(bool showDeleteButton = true, QWidget *parent = nullptr);
    ~LifConversionWidget();

    //! Seed the table from an experiment's already-joined conversion nodes.
    void setFromConfig(const LifConfig &cfg);
    //! Write the table's joined node list into \a cfg.
    void toConfig(LifConfig &cfg) const;

    bool isDirty() const { return d_dirty; }

    LifConversionTableModel *model() const { return p_model; }

signals:
    //! Re-exposed from the model: fired on any wiring/FINAL/harmonic edit.
    void edited();
    //! Re-exposed from the model: a gated harmonic-order change was requested.
    void applyHarmonic(QString stageKey, int n);
    void dirtyChanged(bool dirty);

public slots:
    void clearDirty();
    //! Forwarded to the model: re-reads stageKey's harmonic after a confirmed hardware change.
    void harmonicApplied(const QString &stageKey);

private slots:
    void markDirty();
    void populatePresetCombo();
    void updatePresetBar();
    void onApplyPreset();
    void onSavePreset();
    void onSaveAsPreset();
    void onRenamePreset();
    void onDeletePreset();
    void updatePreview();
    void showTableContextMenu(const QPoint &pos);

private:
    void initializeFromLifPreset(const LifPreset &preset);
    LifPreset toLifPreset() const;
    QString buildChainExpression() const;

    QTableView *p_tableView;
    LifConversionTableModel *p_model;
    QLabel *p_previewLabel;

    QComboBox *p_presetCombo;
    QPushButton *p_applyPresetButton;
    QPushButton *p_savePresetButton;
    QPushButton *p_saveAsPresetButton;
    QPushButton *p_renamePresetButton;
    QPushButton *p_deletePresetButton;

    QIcon d_applyIcon;
    QIcon d_resetIcon;

    bool d_dirty{false};
    bool d_suppressDirty{false};
};

#endif // LIFCONVERSIONWIDGET_H
