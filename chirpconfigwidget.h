#ifndef CHIRPCONFIGWIDGET_H
#define CHIRPCONFIGWIDGET_H

#include <QWidget>
#include "chirptablemodel.h"

namespace Ui {
class ChirpConfigWidget;
}

class ChirpConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChirpConfigWidget(QWidget *parent = 0);
    ~ChirpConfigWidget();

    ChirpConfig getChirpConfig();

public slots:
    void initializeFromSettings();
    void enableEditing(bool enabled);
    void setButtonStates();

    void addSegment();
    void insertSegment();
    void moveSegments(int direction);
    void removeSegments();
    void clear();

    void checkChirp();

signals:
    void chirpConfigChanged(const ChirpConfig);


private:
    Ui::ChirpConfigWidget *ui;
    ChirpTableModel *p_ctm;
    double d_awgMult, d_valonMult, d_txMult, d_txSidebandSign, d_valonFreq;
    QPair<double,double> d_chirpMinMax;

    bool isSelectionContiguous(QModelIndexList l);
    void clearList();


};

#endif // CHIRPCONFIGWIDGET_H
