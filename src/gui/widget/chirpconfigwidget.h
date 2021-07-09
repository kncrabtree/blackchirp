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

namespace BC::Key {
static const QString ChirpConfigWidget("ChirpConfigWidget");
static const QString minPreProt("minPreChirpProtectionUs");
static const QString minPreGate("minPreChirpGateDelayUs");
static const QString minPostProt("minPostChirpGateDelayUs");
static const QString minPostGate("minPostChirpProtectionDelayUs");
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

    void initialize(std::shared_ptr<RfConfig> p);
    void setFromRfConfig();
    QSpinBox *numChirpsBox() const;
    const RfConfig &getRfConfig();

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
    void load();

    void updateChirpPlot();

signals:
    void chirpConfigChanged();


private:
    void updateRfConfig();
    Ui::ChirpConfigWidget *ui;
    ChirpTableModel *p_ctm;
    bool d_rampOnly;
    std::shared_ptr<RfConfig> ps_rfConfig;

    bool isSelectionContiguous(QModelIndexList l);
    void clearList(bool replot=true);



};

#endif // CHIRPCONFIGWIDGET_H
