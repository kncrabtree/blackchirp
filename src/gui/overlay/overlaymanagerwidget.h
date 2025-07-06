#ifndef OVERLAYMANAGERWIDGET_H
#define OVERLAYMANAGERWIDGET_H

#include <QWidget>
#include <QTabWidget>
#include <QToolBar>
#include <QAction>
#include <QVBoxLayout>
#include <QMetaEnum>
#include <QTableView>
#include <QLabel>
#include <QProgressBar>
#include <memory>
#include <map>

#include <data/experiment/overlaybase.h>
#include <data/model/overlaytablemodel.h>
#include <data/storage/overlaystorage.h>
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
    ~OverlayManagerWidget();

signals:
    void overlayAdded(std::shared_ptr<OverlayBase> overlay);
    void overlayRemoved(std::shared_ptr<OverlayBase> overlay);
    void overlayPlotChanged(std::shared_ptr<OverlayBase> overlay, QString newPlotId);
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay);

public slots:
    void addOverlay();
    void removeOverlay();
    void raiseParent();
    
    // Overlay storage event handlers
    void onOverlayWriteCompleted(std::shared_ptr<OverlayBase> overlay);
    void onOverlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error);
    void onPendingWritesChanged(int count);
    
    // Connect to overlay storage signals
    void connectToOverlayStorage(std::shared_ptr<OverlayStorage> storage);

private slots:
    void onModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
    void onSelectionChanged();

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
    
    // Progress indicator widgets
    QLabel *p_progressLabel;
    QProgressBar *p_progressBar;
    QWidget *p_progressWidget;
    
    // Data structure to track model-view pairs for automatic column resizing
    std::map<const OverlayTableModel*, QTableView*> d_modelViewMap;
    
    // Track overlay storage connection
    std::shared_ptr<OverlayStorage> p_overlayStorage;


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
    
    // Progress indicator management
    void createProgressWidget();
    void updateProgressDisplay(int pendingCount);
    void showErrorNotification(const QString& overlayLabel, const QString& error);
};

#endif // OVERLAYMANAGERWIDGET_H
