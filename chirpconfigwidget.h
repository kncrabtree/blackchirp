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

    void updateChirpPlot();

signals:
    void chirpConfigChanged(const ChirpConfig);


private:
    Ui::ChirpConfigWidget *ui;
    ChirpTableModel *p_ctm;
    ChirpConfig d_currentChirpConfig;

    bool isSelectionContiguous(QModelIndexList l);
    void clearList();
    void updateChirpConfig();


};

#endif // CHIRPCONFIGWIDGET_H
