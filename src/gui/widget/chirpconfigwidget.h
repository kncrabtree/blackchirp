#ifndef CHIRPCONFIGWIDGET_H
#define CHIRPCONFIGWIDGET_H

#include <QWidget>

#include <src/data/experiment/rfconfig.h>
#include <src/data/model/chirptablemodel.h>

class QSpinBox;

namespace Ui {
class ChirpConfigWidget;
}

class ChirpConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChirpConfigWidget(QWidget *parent = 0);
    ~ChirpConfigWidget();

    void setRfConfig(const RfConfig c);
    RfConfig getRfConfig();
    QSpinBox *numChirpsBox() const;

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
    Ui::ChirpConfigWidget *ui;
    ChirpTableModel *p_ctm;
    bool d_rampOnly;

    bool isSelectionContiguous(QModelIndexList l);
    void clearList(bool replot=true);



};

#endif // CHIRPCONFIGWIDGET_H
