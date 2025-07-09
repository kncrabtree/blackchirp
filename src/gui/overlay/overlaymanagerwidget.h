#ifndef OVERLAYMANAGERWIDGET_H
#define OVERLAYMANAGERWIDGET_H

#include <QWidget>
#include <QToolBar>
#include <QAction>
#include <QToolButton>
#include <QMenu>
#include <QVBoxLayout>
#include <QTableView>
#include <QLabel>
#include <QProgressBar>
#include <memory>

#include <data/experiment/overlaybase.h>
#include <data/model/overlaytablemodel.h>
#include <data/storage/overlaystorage.h>
#include <data/storage/settingsstorage.h>
#include "overlayconfiguredelegate.h"
#include "overlaycheckboxdelegate.h"


namespace BC::Key::OverlayManager {
static const QString key{"OverlayManagerWidget"};
static const QString geometry{"geometry"};
static const QString windowState{"windowState"};
}

class OverlayManagerWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit OverlayManagerWidget(QWidget *parent = nullptr, int number = -1, const QVector<std::shared_ptr<OverlayBase>> &overlays = QVector<std::shared_ptr<OverlayBase>>());
    ~OverlayManagerWidget();

signals:
    void overlayDataChanged(std::shared_ptr<OverlayBase> overlay);

public slots:
    void addOverlay(OverlayBase::OverlayType type);
    void removeOverlay();
    void raiseParent();
    
    // Overlay storage event handlers
    void onOverlayWriteCompleted(std::shared_ptr<OverlayBase> overlay);
    void onOverlayWriteFailed(std::shared_ptr<OverlayBase> overlay, QString error);
    void onPendingWritesChanged(int count);
    
    // Connect to overlay storage signals
    void connectToOverlayStorage(std::shared_ptr<OverlayStorage> storage);
    
    // Handle external overlay data changes (e.g., from plot context menu)
    void onExternalOverlayDataChanged(std::shared_ptr<OverlayBase> overlay);

private slots:
    void onModelDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);
    void onSelectionChanged();
    void onConfigureClicked(const QModelIndex &index);
    void onOverlaySettingsChanged(std::shared_ptr<OverlayBase> overlay);

private:
    QToolBar *p_toolBar;
    QToolButton *p_addButton;
    QMenu *p_addMenu;
    std::map<OverlayBase::OverlayType, QAction*> d_addActions;
    QAction *p_removeAction;
    QAction *p_raiseParentAction;

    OverlayTableModel *p_overlayModel;
    QTableView *p_overlayTableView;
    OverlayConfigureDelegate *p_configureDelegate;
    OverlayCheckBoxDelegate *p_enabledDelegate;
    
    // Progress indicator widgets
    QLabel *p_progressLabel;
    QProgressBar *p_progressBar;
    QWidget *p_progressWidget;
    
    // Track overlay storage connection
    std::shared_ptr<OverlayStorage> p_overlayStorage;


    void setupUI();
    void setupAddButton();
    void updateButtonStates();
    void populateWithExistingOverlays(const QVector<std::shared_ptr<OverlayBase>> &overlays);
    void setupConfigureDelegate();
    void setupEnabledDelegate();
    void setupTableView();
    void resizeColumnsToContents();
    
    // Progress indicator management
    void createProgressWidget();
    void updateProgressDisplay(int pendingCount);
    void showErrorNotification(const QString& overlayLabel, const QString& error);
    
protected:
    void closeEvent(QCloseEvent *event) override;
};

#endif // OVERLAYMANAGERWIDGET_H
