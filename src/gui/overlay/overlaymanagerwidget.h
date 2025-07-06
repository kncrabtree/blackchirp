#ifndef OVERLAYMANAGERWIDGET_H
#define OVERLAYMANAGERWIDGET_H

#include <QWidget>
#include <QTabWidget>
#include <QToolBar>
#include <QAction>
#include <QVBoxLayout>
#include <QMetaEnum>
#include <QTableView>
#include <memory>
#include <map>

#include <data/experiment/overlaybase.h>
#include <data/model/overlaytablemodel.h>
#include "plotidcomboboxdelegate.h"
#include "overlaynumericdelegate.h"

namespace BC::Property::Overlay {
static const QString overlayType{"overlayType"};
}

class OverlayManagerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit OverlayManagerWidget(QWidget *parent = nullptr, int number = -1, const QVector<std::shared_ptr<OverlayBase>> &overlays = QVector<std::shared_ptr<OverlayBase>>());

signals:
    void overlayAdded(std::shared_ptr<OverlayBase> overlay);
    void overlayRemoved(std::shared_ptr<OverlayBase> overlay);
    void overlayPlotChanged(std::shared_ptr<OverlayBase> overlay, QString newPlotId);
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay);

public slots:
    void addOverlay();
    void removeOverlay();
    void raiseParent();

private slots:
    void onModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

private:
    QTabWidget *p_tabWidget;
    QToolBar *p_toolBar;
    QAction *p_addAction;
    QAction *p_removeAction;
    QAction *p_raiseParentAction;

    BCExperimentOverlayModel *p_bcExperimentModel;
    QTableView *p_bcExperimentTableView;
    PlotIdComboBoxDelegate *p_plotIdDelegate;
    OverlayNumericDelegate *p_numericDelegate;
    
    // Data structure to track model-view pairs for automatic column resizing
    std::map<const OverlayTableModel*, QTableView*> d_modelViewMap;


    void setupUI();
    void createTabs();
    QWidget* createBCExperimentTab();
    QWidget* createPlaceholderTab(const QString& tabName);
    void onTabChanged(int index);
    void updateButtonStates();
    void populateWithExistingOverlays(const QVector<std::shared_ptr<OverlayBase>> &overlays);
    void setupPlotIdDelegate();
    void setupTableView();
    void resizeColumnsToContents(const OverlayTableModel* model, QTableView* tableView);
};

#endif // OVERLAYMANAGERWIDGET_H
