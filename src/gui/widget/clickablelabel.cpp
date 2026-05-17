#include <gui/widget/clickablelabel.h>

#include <QEnterEvent>
#include <QMouseEvent>
#include <QStyle>
#include <QDesktopServices>
#include <QUrl>

ClickableLabel::ClickableLabel(QWidget *parent) : QLabel(parent)
{
    // Needed so hover state can track whether the pointer is over the
    // text region rather than just the (full-width) widget.
    setMouseTracking(true);
}

void ClickableLabel::setFolderPath(const QString &path)
{
    d_folderPath = path;
    if(d_folderPath.isEmpty())
    {
        d_hot = false;
        unsetCursor();
        setUnderline(false);
        setToolTip(QString());
    }
    else
        setToolTip(d_folderPath);
}

QRect ClickableLabel::textRect() const
{
    QFont f = font();
    f.setUnderline(false);
    const QFontMetrics fm(f);
    const QSize ts = fm.size(Qt::TextSingleLine, text())
                         .boundedTo(contentsRect().size());
    return QStyle::alignedRect(layoutDirection(), alignment(),
                               ts, contentsRect());
}

void ClickableLabel::updateHover(const QPoint &pos)
{
    const bool over = !d_folderPath.isEmpty() && textRect().contains(pos);
    if(over == d_hot)
        return;
    d_hot = over;
    if(over)
    {
        setCursor(Qt::PointingHandCursor);
        setUnderline(true);
    }
    else
    {
        unsetCursor();
        setUnderline(false);
    }
}

void ClickableLabel::enterEvent(QEnterEvent *e)
{
    updateHover(e->position().toPoint());
    QLabel::enterEvent(e);
}

void ClickableLabel::mouseMoveEvent(QMouseEvent *e)
{
    updateHover(e->position().toPoint());
    QLabel::mouseMoveEvent(e);
}

void ClickableLabel::leaveEvent(QEvent *e)
{
    if(d_hot)
    {
        d_hot = false;
        unsetCursor();
        setUnderline(false);
    }
    QLabel::leaveEvent(e);
}

void ClickableLabel::mouseReleaseEvent(QMouseEvent *e)
{
    if(!d_folderPath.isEmpty() && e->button() == Qt::LeftButton
       && textRect().contains(e->position().toPoint()))
        QDesktopServices::openUrl(QUrl::fromLocalFile(d_folderPath));
    QLabel::mouseReleaseEvent(e);
}

void ClickableLabel::setUnderline(bool on)
{
    QFont f = font();
    f.setUnderline(on);
    setFont(f);
}
