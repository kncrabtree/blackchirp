#ifndef FTMWCONFIGWIDGET_H
#define FTMWCONFIGWIDGET_H

#include <QHash>
#include <QIcon>
#include <QWidget>

#include <data/experiment/rfconfig.h>
#include <data/loadout/hardwareloadout.h>
#include <data/settings/guikeys.h>
#include <data/storage/settingsstorage.h>

class QComboBox;
class QGroupBox;
class QLabel;
class QPushButton;
class QTabWidget;
class RfConfigWidget;
class ChirpConfigWidget;
class FtmwDigitizerConfigWidget;
class FtmwConfig;
class LoadoutManager;

class FtmwConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit FtmwConfigWidget(const QString &awgHwKey, const QString &digiHwKey,
                              const QHash<RfConfig::ClockType, RfConfig::ClockFreq> &currentClocks,
                              bool showDeleteButton = true,
                              QWidget *parent = nullptr);

    void initializeFromFtmwPreset(const FtmwPreset &preset);
    FtmwPreset toFtmwPreset() const;
    void initializeFromExperiment(const FtmwConfig &cfg);
    void updateChirpFromRf();

    bool isDirty() const { return d_dirty; }

    void setTabError(int tabIndex, bool hasError);
    void clearTabErrors();

    RfConfigWidget *rfConfigWidget() const { return p_rfWidget; }
    ChirpConfigWidget *chirpConfigWidget() const { return p_chirpWidget; }
    FtmwDigitizerConfigWidget *digiWidget() const { return p_digiWidget; }

signals:
    void applyClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq> clocks);
    void dirtyChanged(bool dirty);

public slots:
    void clearDirty();

private slots:
    void markDirty();
    void populatePresetCombo();
    void updatePresetBar();
    void onApplyPreset();
    void onSavePreset();
    void onSaveAsPreset();
    void onRenamePreset();
    void onDeletePreset();

private:
    void populateSourceCombos();
    void onRfSourceChanged(int index);
    void onChirpSourceChanged(int index);
    void onDigiSourceChanged(int index);

    QTabWidget *p_tabWidget;
    QComboBox *p_rfSourceCombo;
    QComboBox *p_chirpSourceCombo;
    QComboBox *p_digiSourceCombo;
    RfConfigWidget *p_rfWidget;
    ChirpConfigWidget *p_chirpWidget;
    FtmwDigitizerConfigWidget *p_digiWidget;

    QComboBox *p_ftmwPresetCombo;
    QPushButton *p_applyPresetButton;
    QPushButton *p_savePresetButton;
    QPushButton *p_saveAsPresetButton;
    QPushButton *p_renamePresetButton;
    QPushButton *p_deletePresetButton;

    QIcon d_applyIcon;
    QIcon d_resetIcon;

    QString d_awgHwKey;
    QString d_digiHwKey;
    bool d_dirty = false;
    bool d_suppressDirty = false;

    friend class LoadoutManager;
};

#endif // FTMWCONFIGWIDGET_H
