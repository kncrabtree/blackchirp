#ifndef OVERLAYMANAGERWIDGET_H
#define OVERLAYMANAGERWIDGET_H

#include <QWidget>
#include <QTabWidget>
#include <QToolBar>
#include <QAction>
#include <QVBoxLayout>
#include <QMetaEnum>
#include <QTableView>

#include <data/experiment/overlaybase.h>
#include <data/model/overlaytablemodel.h>

namespace BC::Property::Overlay {
static const QString overlayType{"overlayType"};
}

class OverlayManagerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit OverlayManagerWidget(QWidget *parent = nullptr, int number = -1);

signals:
    void overlayAdded(OverlayBase* overlay);
    void overlayRemoved(OverlayBase* overlay);

public slots:
    void addOverlay();
    void removeOverlay();
    void raiseParent();

private:
    QTabWidget *p_tabWidget;
    QToolBar *p_toolBar;
    QAction *p_addAction;
    QAction *p_removeAction;
    QAction *p_raiseParentAction;

    BCExperimentOverlayModel *p_bcExperimentModel;
    QTableView *p_bcExperimentTableView;


    void setupUI();
    void createTabs();
    QWidget* createBCExperimentTab();
    QWidget* createPlaceholderTab(const QString& tabName);
    void onTabChanged(int index);
    void updateButtonStates();
};

#endif // OVERLAYMANAGERWIDGET_H
