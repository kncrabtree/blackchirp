#ifndef FTMWSNAPSHOTWIDGET_H
#define FTMWSNAPSHOTWIDGET_H

#include <QWidget>

#include "fid.h"

class QThread;
class QListWidget;
class QSpinBox;
class QPushButton;
class QRadioButton;
class SnapWorker;

class FtmwSnapshotWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FtmwSnapshotWidget(int num, const QString path = QString(""), QWidget *parent = 0);
    ~FtmwSnapshotWidget();

    int count() const;
    bool isEmpty() const;

    QSize sizeHint() const;
    void setFidList(const FidList l);

signals:
    void loadFailed(const QString errMsg);
    void snapListChanged();
    void finalizedList(const FidList);
    void experimentLogMessage(int,QString,BlackChirp::LogMessageCode=BlackChirp::LogNormal,QString=QString(""));

public slots:
    void setSelectionEnabled(bool en);
    void setFinalizeEnabled(bool en);
    bool readSnapshots();
    void updateSnapList();
    void snapListUpdated(const FidList l);
    void finalize();

private:
    QListWidget *p_lw;
    QRadioButton *p_allButton, *p_recentButton, *p_selectedButton;
    QPushButton *p_finalizeButton, *p_selectAllButton, *p_selectNoneButton;
    QThread *p_workerThread;
    SnapWorker *p_sw;

    int d_num;
    bool d_busy, d_updateWhenDone;
    QString d_path;
};

#endif // FTMWSNAPSHOTWIDGET_H
