#ifndef CHIRPCONFIGWIDGET_H
#define CHIRPCONFIGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/rfconfig.h>
#include <data/model/chirptablemodel.h>

#include <memory>

class QSpinBox;

namespace Ui {
class ChirpConfigWidget;
}

namespace BC::Key::ChirpConfigWidget {
static const QString key{"ChirpConfigWidget"};
static const QString minPreProt{"minPreChirpProtectionUs"};
static const QString minPreGate{"minPreChirpGateDelayUs"};
static const QString minPostProt{"minPostChirpGateDelayUs"};
static const QString minPostGate{"minPostChirpProtectionDelayUs"};
static const QString preProt{"preChirpProtectionUs"};
static const QString postProt{"postChirpProtectionUs"};
static const QString preGate{"preChirpGateUs"};
static const QString postGate{"postChirpGateUs"};
static const QString numChirps{"numChirps"};
static const QString interval{"chirpIntervalUs"};
static const QString applyAll{"applyToAll"};
}

/*!
 * \brief The ChirpConfigWidget class
 *
 * The post chirp gate and post chirp protection are both measured with respect to the end of the chirp!
 *
 */
class ChirpConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit ChirpConfigWidget(QWidget *parent = 0);
    ~ChirpConfigWidget();

    void initialize(const RfConfig &rfc);
    void setFromRfConfig(const RfConfig &rfc);
    ChirpConfig &getChirps();

public slots:
    void enableEditing(bool enabled);
    void setButtonStates();

    void addSegment();
    void addEmptySegment();
    void insertSegment();
    void insertEmptySegment();
    void moveSegments(int direction);
    void removeSegments();
    void clear();

    void updateChirpPlot();

signals:
    void chirpConfigChanged();


private:
    void updateRfConfig();
    Ui::ChirpConfigWidget *ui;
    ChirpTableModel *p_ctm;
    bool d_rampOnly;
    double d_awgSampleRate;
    RfConfig d_rfConfig;

    bool isSelectionContiguous(QModelIndexList l);
    void clearList(bool replot=true);



};

#endif // CHIRPCONFIGWIDGET_H
