\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{hnuthesis}
    [2017/07/19 v0.1 HNU thesis template]

% 变量
\newif\if@hnu@doctor
\newif\if@hnu@numerical
\newif\if@hnu@super
\newif\if@hnu@pdf
\DeclareOption{doctor}{\@hnu@doctortrue}
\DeclareOption{print}{\@hnu@pdffalse}
\DeclareOption{pdf}{\@hnu@pdftrue}
\DeclareOption{super}{\@hnu@numericaltrue\@hnu@supertrue}
\DeclareOption{numbers}{\@hnu@numericaltrue\@hnu@superfalse}
\DeclareOption{authoryear}{\@hnu@numericalfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ExecuteOptions{doctor,chinese,print,super}
\ProcessOptions\relax

\newif\if@hnu@addtocspace
\if@hnu@pdf
  \PassOptionsToClass{oneside}{book}
\fi
\PassOptionsToPackage{no-math}{fontspec}
\LoadClass[UTF8, a4paper, openany, zihao=-4]{ctexbook}[2015/05/14]
\RequirePackage{etoolbox}
\@ifclasslater{ctexbook}{2015/05/14}{}{%
    \ClassError{hnuthesis}{%
        Requiring ctex 2.0 or later version!\MessageBreak
        Please update the package using your\MessageBreak
        TeX package manager or from CTAN
    }{%
        See logs for more details.
    }
}
%字体设置
% [本地修改 1/2] 西文字体：优先使用模板指定的 Times New Roman / Arial / Courier New；
% 系统未安装时（如 Linux）退回到 TeX 发行版自带、字形度量相近的 TeX Gyre 字体。
% 在装有上述字体的系统（Windows）上编译，结果与原模板一致。
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{%
  \setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic]}
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{%
  \setsansfont{texgyreheros}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic]}
\IfFontExistsTF{Courier New}{\setmonofont{Courier New}}{%
  \setmonofont{texgyrecursor}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic]}
% [本地修改 2/2] 随模板分发的宋体与黑体按文件名加载：新版 fontspec 不再接受原模板的 [simsun.ttc] 写法。
\setCJKmainfont{simsun.ttc}
\setCJKsansfont{simhei.ttf}

\newcommand{\yihao}{\fontsize{26pt}{39pt}\selectfont}
\newcommand{\xiaoyi}{\fontsize{24pt}{36pt}\selectfont}   
\newcommand{\erhao}{\fontsize{22pt}{33pt}\selectfont}          
\newcommand{\xiaoer}{\fontsize{18pt}{27pt}\selectfont}          
\newcommand{\sanhao}{\fontsize{16pt}{24pt}\selectfont}        
\newcommand{\xiaosan}{\fontsize{15pt}{22.5pt}\selectfont}        
\newcommand{\sihao}{\fontsize{14pt}{21pt}\selectfont}            
\newcommand{\xiaosi}{\fontsize{12pt}{18pt}\selectfont}            
\newcommand{\wuhao}{\fontsize{10.5pt}{15.75pt}\selectfont}
\newcommand{\xiaowu}{\fontsize{9pt}{13.5pt}\selectfont}    
\newcommand{\liuhao}{\fontsize{7.5pt}{11.25pt}\selectfont}


\RequirePackage{hyperref}
\hypersetup{
    bookmarksopen=true,
    bookmarksnumbered=true,
    bookmarksopenlevel=1,
    CJKbookmarks=true,
    pdfborder=0 0 0,
    unicode=true,
    linktoc=all,
}
\if@hnu@pdf
    \hypersetup{
        colorlinks=true,
        allcolors=blue,
    }
\fi

\AtBeginDocument{
    \hypersetup{
        pdftitle={\hnu@title},
        pdfauthor={\hnu@author}
    }
}

\RequirePackage{xparse}
\NewDocumentCommand\setfontsize{mo}{\IfValueTF{#2}{\fontsize{#1}{#2}}{\fontsize{#1}{2\dimexpr#1}}\linespread{1}\selectfont\relax}
\newcommand\zhspace[1][1]{\hspace{#1\ccwd}}
\newcommand\hnu@underline[2][6em]{%
    \hskip1pt\underline{\hb@xt@ #1{\hss#2\hss}}\hskip3pt\relax
}
\def\hnu@define@term#1{
    \expandafter\gdef\csname #1\endcsname##1{
        \expandafter\gdef\csname hnu@#1\endcsname{##1}
    }
    \csname #1\endcsname{}
}
\hnu@define@term{hnucode}
\hnu@define@term{hnuname}
\hnu@define@term{title}
\hnu@define@term{clc}
\hnu@define@term{secrettext}
\hnu@define@term{author}
\hnu@define@term{authorid}
\hnu@define@term{college}
\hnu@define@term{major}
\hnu@define@term{workon}
\hnu@define@term{supervisor}
\hnu@define@term{cosupervisor}
\hnu@define@term{submitdate}
\hnu@define@term{defensedate}
\hnu@define@term{chair}
\hnu@define@term{entitle}
\hnu@define@term{enauthor}
\hnu@define@term{enmajor}
\hnu@define@term{endoctor}
\hnu@define@term{enmaster}
\hnu@define@term{enbachelor}
\hnu@define@term{enhnuname}
\hnu@define@term{ensupervisor}
\hnu@define@term{endate}

\if@hnu@doctor
  \newcommand\hnu@thesisname{硕士专业学位研究生学位论文}%
\fi

% 表格：
% 三线表
\RequirePackage{booktabs}
\RequirePackage{multirow}
% 跨页表格
% \RequirePackage{longtable}



% 题注、标题
\newcommand\hnu@notesname{\textbf{注}：}
\newcommand\hnu@enabstractname{Abstract}
\newcommand\hnu@abstractname{摘\qquad 要}
\newcommand\hnu@acknowledgementsname{致\qquad 谢}
\newcommand\hnu@publicationsname{在读期间公开发表论文（著）及科研情况}

\newcommand\hnu@tocname{目\qquad 录}
\newcommand\hnu@summaryname{结\qquad 语}
\newcommand\hnu@notationname{符号说明}
\def\equationautorefname~#1\null{公式~(#1)\null}%
\def\footnoteautorefname{脚注}%
\def\itemautorefname~#1\null{第~#1~项\null}%
\def\figureautorefname{图}%
\def\tableautorefname{表}%
\def\partautorefname~#1\null{第~#1~部分\null}%
\def\appendixautorefname{}%
\def\chapterautorefname~#1\null{第~#1~章\null}%
\def\sectionautorefname~#1\null{第~#1~节\null}%
\def\subsectionautorefname~#1\null{第~#1~小节\null}%
\def\subsubsectionautorefname~#1\null{第~#1~小小节\null}%
\def\paragraphautorefname~#1\null{第~#1~段\null}%
\def\subparagraphautorefname~#1\null{第~#1~小段\null}%
\def\theoremautorefname{定理}%
\def\pageautorefname~#1\null{第~#1~页\null}%

% 页面设置
\RequirePackage{geometry}
\geometry{
    paper=a4paper,
    top=2.54cm, bottom=2.54cm,
    left=3.18cm, right=3.18cm,
    headheight=0.4cm, headsep=0.85cm,
    footskip=1.1cm}
\RequirePackage{fancyhdr}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{hnu@headings}{%
  \fancyhf{}%
  \fancyhead[CE]{\setfontsize{9pt}\hnu@title}%
  \fancyhead[CO]{\setfontsize{9pt}\hnu@thesisname}
  \fancyfoot[C]{\setfontsize{9pt}\thepage}
  \fancyheadoffset[LO,LE]{3mm}
  \fancyheadoffset[RO,RE]{-3mm}
  \makeatletter
%   \def\headrule{
%     \if@fancyplain\let\headrulewidth\plainheadrulewidth\fi%
%     \hrule\@height 1.0pt \@width\headwidth\vskip 1pt %上面线为1pt粗
%     \hrule\@height 0.5pt \@width\headwidth  %下面0.5pt粗
%     \vskip-2\headrulewidth\vskip-1pt}      %两条线的距离1pt
%     \vspace{7mm}
}%

\fancypagestyle{hnu@notation}{\fancyfoot{}}
\pagestyle{hnu@headings}
\patchcmd\chaptermark{#1}{\protect\spacetitle{#1}}{}{}
\renewcommand*{\cleardoublepage}{\clearpage\if@twoside \ifodd\c@page\else
  \hbox{}\thispagestyle{empty}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\g@addto@macro\frontmatter{%
  \pagenumbering{Roman}%
}
\renewcommand\mainmatter{%
  \cleardoublepage
  \pagenumbering{arabic}%
  \@mainmattertrue
}

% 列表
\RequirePackage{enumitem}
\setitemize{leftmargin=3em,itemsep=0em,partopsep=0em,parsep=0em,topsep=-0em}
\setenumerate{leftmargin=3em,itemsep=0em,partopsep=0em,parsep=0em,topsep=0em}

% 章节、段落、页眉、页脚
\renewcommand\chapter{%
  \if@openright\cleardoublepage\else\clearpage\fi
  \setlength{\baselineskip}{20pt}
  \renewcommand{\CJKglue}{\hskip 0.5pt plus \baselineskip}
  \global\@topnum\z@
  \@afterindenttrue
  \secdef\@chapter\@schapter
}

\RequirePackage{pifont}
\def\hnu@textcircled#1{%
  \ifnum\value{#1} >9
    \ClassError{hnuthesis}%
      {Too many footnotes in this page.}{Keep footnote less than 10.}
  \fi
  \ding{\the\numexpr\value{#1}+191\relax}
}

\RequirePackage{stringstrings}
\newcounter{titlelength}
\newcommand\spacetitle[1]{%
  \@getstringlength{#1}{titlelength}%
  \ifnum\thetitlelength=2{\ziju{2}#1}\else%
    \ifnum\thetitlelength=3{\ziju{1}#1}\else%
      \ifnum\thetitlelength=4{\ziju{0.5}#1}\else#1%
      \fi
    \fi
  \fi
}
\setcounter{secnumdepth}{5}
\ctexset{
    chapter = {
        format      = \centering\sffamily\heiti\bfseries\setfontsize{16pt},
        nameformat  = {},
        % titleformat = \spacetitle,
        number      = \thechapter,
        aftername   = \hspace{\ccwd},
        beforeskip  = 12pt, % 24bp - 31bp %1em
        afterskip   = 12pt, % 18bp - 10bp
        % fixskip   = true, % will be used in future version
    },
    section = {
        format     = \bf\sffamily\heiti\setfontsize{15pt},
        aftername  = \hspace{\ccwd},
        beforeskip = 0pt,
        afterskip  = 0pt,
    },
    subsection = {
        format     = \bf\sffamily\heiti\setfontsize{14pt},
        aftername  = \hspace{\ccwd},
        % indent     = 2\ccwd,
        beforeskip = 2pt,
        afterskip  = 2pt,
    },
    subsubsection = {
        format    = \sffamily\heiti\setfontsize{12pt},
        number    = (\arabic{subsubsection}),
        % aftername = \hspace{\ccwd},
        beforeskip = 2pt,
        afterskip  = 2pt,
        % indent    = 3\ccwd,
    },
    paragraph = {
        format    = \sffamily\setfontsize{12pt},
        number    = (\arabic{paragraph}),
        aftername = \hspace{\ccwd},
        % indent    = 3\ccwd,
    },
    subparagraph = {
        format    = \sffamily\setfontsize{12pt},
        number    = \hnu@textcircled{subparagraph},
        aftername = \hspace{\ccwd},
        % indent    = 3\ccwd,
    },
}

% 书签
\newcounter{hnu@pdfbookmark}
\NewDocumentCommand\hnu@chapter{o m}{%
  \if@openright\cleardoublepage\else\clearpage\fi%
  \addtocounter{hnu@pdfbookmark}\@ne
  \IfValueTF{#1}{%
    \pdfbookmark[0]{#1}{hnuchapter.\thehnu@pdfbookmark}%
    \chaptermark{#1}%
  }{%
    \pdfbookmark[0]{#2}{hnuchapter.\thehnu@pdfbookmark}%
    \chaptermark{#2}%
  }%
  \chapter*{#2}
}

% 封面
\renewcommand\maketitle{%
    \newgeometry{
        top=3.8cm, bottom=3.8cm,
        left=3.2cm, right=3.2cm,
        headheight=0cm, headsep=0.8cm,
        footskip=0.8cm}
    \pagenumbering{gobble}
    \pdfbookmark[0]{封面}{titlepage}
    \make@cntitle
    % \pdfbookmark[0]{Title page}{entitlepage}
    \restoregeometry
    \pdfbookmark[0]{原创性声明}{statement}
    \make@statement
}

\newcommand\hnu@supervisorline{%
  \ifx\hnu@encosupervisor\@empty%
    Supervisor: \hnu@ensupervisor%
  \else%
    Supervisors: \hnu@ensupervisor, \hnu@encosupervisor%
  \fi}

\RequirePackage{tikz}
\newcommand\vpostext[2]{%
  \tikz[remember picture,overlay]%
      \node [yshift=-#1] at (current page.north) [below,align=flush center]%
          {\parbox{\textwidth}{\centering#2}};
}

\newcommand\linetext[3]{%
  \underline{\makebox[#1][#2]{#3}}
}


% 中文封面
\RequirePackage{tabularx}
\newcommand\make@cntitle{%
  \begin{titlepage}%
     \tikz[remember picture,overlay]%
        \node [xshift=2.5cm,yshift=-1.5cm] at (current page.north west)%
            [below right,align=left] {%

          \newlength{\@tw}
          \setlength{\@tw}{55pt}
          \newlength{\@cw}
          \setlength{\@cw}{3.2cm}
          
          \setfontsize{12pt}%
          \begin{tabularx}{\linewidth}{lcccr}%
            \makebox[\@tw][s]{\sffamily\songti{分类号}} & \linetext{\@cw}{c}{\hnu@clc} &
            \makebox[2.5cm][c]{\qquad} &
            \makebox[\@tw][s]{\sffamily\songti{密级}} & \linetext{\@cw}{c}{\hnu@secrettext} \\%
            \makebox[\@tw][s]{\sffamily\songti{学校代号}} & \linetext{\@cw}{c}{\hnu@hnucode} &
            \makebox[2.5cm][c]{\qquad} &
            \makebox[\@tw][s]{\sffamily\songti{学号}} & \linetext{\@cw}{c}{\hnu@authorid}
            
          \end{tabularx}};%

      \vpostext{3.5cm}{\includegraphics[height=4.5cm]{figures/Jxnu-logo.png}}%
      \vpostext{8.5cm}{\sffamily\setfontsize{26pt}\hnu@thesisname}%
      \vpostext{11.5cm}{\bf\sffamily\bfseries\kaishu\xiaoyi{中文题目：}\hnu@title}%
      \vpostext{14.7cm}{\bf\sffamily\kaishu\xiaoer{英文题目：}\hnu@entitle}%
      \vpostext{17.5cm}{\bf\sffamily\kaishu\xiaoer{学生姓名：}\hnu@author}%

     \tikz[remember picture,overlay]%
        \node [xshift=2.5cm,yshift=-21cm] at (current page.north west)%
            [below right,align=left] {%

          \newlength{\@mw}
          \setlength{\@mw}{55pt}
          \newlength{\@sw}
          \setlength{\@sw}{3.2cm}

          \setfontsize{16pt}%
          \begin{tabularx}{\linewidth}{lcccr}%
            \makebox[\@mw][s]{\sffamily\kaishu{院}\qquad{所：}} &{\hnu@college} & {\quad}
            \makebox[\@mw][s]{\sffamily\kaishu{导师姓名：}} & {\quad\hnu@supervisor} \\
            \makebox[\@mw][s]{\sffamily\kaishu{专业学位类别：}} & {\qquad\quad\hnu@major} & {\quad}
            \makebox[\@mw][s]{\sffamily\kaishu{专业领域：}} & {\quad\hnu@workon}
          \end{tabularx}};%
      
      \vpostext{25cm}{\sffamily\kaishu\setfontsize{22pt}\hnu@enbachelor}
      
      \clearpage
      \thispagestyle{empty}

  \end{titlepage}
}


% 声明
\newcommand{\hnu@declaretext} {\fangsong\setfontsize{14pt}
本人声明所呈交的学位论文是本人在导师指导下进行的研究工作及取得的研究成果。据我所知，除了文中特别加以标注和致谢的地方外，论文中不包含其他人已经发表或撰写过的研究成果，也不包含为获得或其他教育机构的学位或证书而使用过的材料。与我一同工作的同志对本研究所做的任何贡献均已在论文中作了明确的说明并表示谢意。
}

\newcommand{\hnu@authorization}{\fangsong\setfontsize{14pt}
本学位论文作者完全了解江西师范大学研究生院有关保留、使用学位论文的规定，有权保留并向国家有关部门或机构送交论文的复印件和磁盘，允许论文被查阅和借阅。本人授权江西师范大学研究生院可以将学位论文的全部或部分内容编入有关数据库进行检索，可以采用影印、缩印或扫描等复制手段保存、汇编学位论文。

（保密的学位论文在解密后适用本授权书）

}

\newcommand\make@statement{
    \newgeometry{left=3.18cm,right=3.18cm,top=2.54cm, bottom=2.54cm}
    \thispagestyle{plain}
    \pagenumbering{Roman}
    % \setcounter{page}{1}
    %\addcontentsline{toc}{chapter}{独创性声明和学位论文版权使用授权书}
    %\setfontsize{12.5pt}[20pt]
    \vspace*{-0.15cm}
    \begin{center}
      \sffamily\fangsong
     \setfontsize{18pt}\bfseries{独创性声明}
    \end{center}
    \vskip 0.3cm

    \hnu@declaretext\par

    \vskip 1cm
    {\fangsong\setfontsize{14pt}学位论文作者签名：\qquad\qquad\qquad\ 签字日期：\qquad 年\qquad 月\qquad 日}
    \vskip 1cm

    \begin{center}
      \sffamily\fangsong
      \setfontsize{16pt}\textbf{学位论文版权使用授权书}
    \end{center}
    \vskip 0.3cm

    \hnu@authorization\par
    \vskip 1cm
    {\fangsong\setfontsize{14pt}
    学位论文作者签名：\qquad\qquad\qquad  导师签名：
    
    签字日期：\qquad 年\quad 月\quad 日\qquad\quad  签字日期：\qquad 年\quad 月\quad 日}
    \restoregeometry
}

% 摘要、关键字
\newenvironment{abstract}{%
    \hnu@chapter{\hnu@abstractname}%
    \addcontentsline{toc}{chapter}{\hnu@abstractname}
    \setcounter{page}{1}
  }{}
\newenvironment{enabstract}{%
    \hnu@chapter[Abstract]{\hnu@enabstractname}
    \addcontentsline{toc}{chapter}{\hnu@enabstractname}
  }{}

\newenvironment{publications}{%
    \hnu@chapter{\hnu@publicationsname}%
    \addcontentsline{toc}{chapter}{\hnu@publicationsname}
}{}

\newenvironment{summary}{%
    \hnu@chapter{\hnu@summaryname}%
    \addcontentsline{toc}{chapter}{\hnu@summaryname}
}{}
\newenvironment{acknowledgements}{%
  \chapter{\hnu@acknowledgementsname}%
%   \addcontentsline{toc}{chapter}{\hnu@acknowledgementsname}
}{}

\newcommand\keywords[1]{%
  \par\phantom{empty}\par\noindent\hangindent=4\ccwd\relax%
  \sffamily\quad\textbf{\setfontsize{12pt}\quad 关键词}\setfontsize{12pt}：#1}
\newcommand\enkeywords[1]{%
  \par\phantom{empty}\par\noindent\hangindent=5.3em\relax%
  \quad\textbf{\setfontsize{12pt}\quad Key Words}\setfontsize{12pt}: #1}

% 目录
\renewcommand\tableofcontents{%
    \clearpage%
    \songti\setfontsize{12pt}\setcounter{tocdepth}{2}
    \addcontentsline{toc}{chapter}{\hnu@tocname}
    \hnu@chapter{\hnu@tocname}%
    \@starttoc{toc}
}

\RequirePackage{titletoc}
\newcommand\hnu@leaders{\titlerule*[0.5pc]{$\cdot$}}
  \titlecontents{chapter}[0bp]
    {\addvspace{6bp}\bf\sffamily\setfontsize{12.5pt}[15pt]}
    {\contentspush{\thecontentslabel\unskip\hskip\ccwd\relax}}
    {}{\hnu@leaders\setfontsize{12.5pt}[12.5pt]\contentspage}
  \titlecontents{section}
      [2\ccwd]
      {\setfontsize{12.5pt}[20pt]}
      {\contentspush{\thecontentslabel\hskip\ccwd\relax}}
      {}{\hnu@leaders\setfontsize{12.5pt}[12.5pt]\contentspage}
  \titlecontents{subsection}
      [4\ccwd]
      {\setfontsize{12.5pt}[20pt]}
      {\contentspush{\thecontentslabel\hskip\ccwd\relax}}
      {}{\hnu@leaders\setfontsize{12.5pt}[12.5pt]\contentspage}

  \titlecontents{figure}[0bp]
      {\setfontsize{12.5pt}}
      {\figureautorefname\hspace*{0.5em}\thecontentslabel\quad}
      {}{\hnu@leaders\contentspage}
  \titlecontents{table}[0bp]
      {\setfontsize{12.5pt}}
      {\tableautorefname\hspace*{0.5em}\thecontentslabel\quad}
      {}{\hnu@leaders\contentspage}
      % 为了让图目录另页起，重新定义 \cs{listoffigures}
\renewcommand\listoffigures{%
    \clearpage
    \hnu@chapter{\listfigurename}%
    \addcontentsline{toc}{chapter}{\listfigurename}
    \@starttoc{lof}}
    % 表目录同样
\renewcommand\listoftables{%
    \clearpage
    \hnu@chapter{\listtablename}%
    \addcontentsline{toc}{chapter}{\listtablename}
    \@starttoc{lot}}

\newenvironment{notation}{%
    \cleardoublepage
    \thispagestyle{hnu@notation}
    \ctexset{chapter/format = \centering\rmfamily\setfontsize{12bp}}
    \hnu@chapter{\hnu@notationname}
    \setfontsize{10.5bp}[16bp]
    \setlength{\itemsep}{0bp}}{}
\newdimen\bp@ \bp@=1bp

\renewcommand\normalsize{%
  \@setfontsize\normalsize{12\bp@}{20\bp@}%
  \abovedisplayskip 12\bp@ \@plus3\bp@ \@minus7\bp@
  \abovedisplayshortskip \z@ \@plus3\bp@
  \belowdisplayshortskip 6.5\bp@ \@plus3.5\bp@ \@minus3\bp@
  \belowdisplayskip \abovedisplayskip
  \let\@listi\@listI}
\normalsize
  \renewcommand\small{%
     \@setfontsize\small{10.5\bp@}{17.5\bp@}%
     \abovedisplayskip 10.5\bp@ \@plus3\bp@ \@minus6\bp@
     \abovedisplayshortskip \z@ \@plus3\bp@
     \belowdisplayshortskip 6.5\bp@ \@plus3.5\bp@ \@minus3\bp@
     \def\@listi{\leftmargin\leftmargini
                 \topsep 9\bp@ \@plus3\bp@ \@minus5\bp@
                 \parsep 4.5\bp@ \@plus2\bp@ \@minus\bp@
                 \itemsep \parsep}%
     \belowdisplayskip \abovedisplayskip
  }
  \renewcommand\footnotesize{%
     \@setfontsize\footnotesize{9\bp@}{15\bp@}
     \abovedisplayskip 9\bp@ \@plus2\bp@ \@minus5\bp@
     \abovedisplayshortskip \z@ \@plus3\bp@
     \belowdisplayshortskip 6\bp@ \@plus3\bp@ \@minus3\bp@
     \def\@listi{\leftmargin\leftmargini
                 \topsep 6\bp@ \@plus2\bp@ \@minus2\bp@
                 \parsep 3\bp@ \@plus2\bp@ \@minus\bp@
                 \itemsep \parsep}%
     \belowdisplayskip \abovedisplayskip
  }
  \renewcommand\scriptsize{\@setfontsize\scriptsize{7.5\bp@}{12.5\bp@}}
  \renewcommand\tiny{\@setfontsize\tiny{6.5\bp@}{10.83\bp@}}
  \renewcommand\large{\@setfontsize\large{15\bp@}{25\bp@}}
  \renewcommand\Large{\@setfontsize\Large{18\bp@}{30\bp@}}
  \renewcommand\LARGE{\@setfontsize\LARGE{22\bp@}{36.67\bp@}}
  \renewcommand\huge{\@setfontsize\huge{24\bp@}{40\bp@}}
  \renewcommand\Huge{\@setfontsize\Huge{26\bp@}{43.33\bp@}}

% 排版风格，标题，图表
\setlength{\parindent}{2\ccwd}
\setlength{\parskip}{\z@}
\RequirePackage{upgreek}
\renewcommand\pi{\uppi}
\RequirePackage{amssymb}
\renewcommand {\thetable} {\thechapter{}-\arabic{table}}
\renewcommand {\thefigure} {\thechapter{}-\arabic{figure}}
\renewcommand\le{\leqslant}
\renewcommand\leq{\leqslant}
\renewcommand\ge{\geqslant}
\renewcommand\geq{\geqslant}
\DeclareMathSizes{10.5bp}{10.5bp}{7.35bp}{5.25bp}
\renewcommand{\topfraction}{.85}
\renewcommand{\bottomfraction}{.65}
\renewcommand\textfraction{.15}
\renewcommand{\floatpagefraction}{.6}
\RequirePackage{caption}
\RequirePackage{calc}
\DeclareCaptionLabelSeparator{zhspace}{\hspace{\ccwd}}
\captionsetup{
    format = hang,
    font = {small,bf},
    labelsep = zhspace,
}
\captionsetup[figure]{
    position = bottom,
    aboveskip = 6bp,
    belowskip = {12bp-\intextsep},
}
\captionsetup[table]{
    position = top,
    aboveskip = 6bp,
    belowskip = 6bp,
}
\newcommand\note[1]{%
    \captionsetup{position = bottom, font = small}
    \caption*{\hangindent=2\ccwd\relax\hnu@notesname\rmfamily#1}}
\renewcommand{\thefootnote}{\hnu@textcircled{footnote}}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.25\textwidth
  \kern2.6\p@}
\renewcommand\@makefntext[1]{%
  \parindent 2\ccwd%
  \noindent
  \hb@xt@2\ccwd{\hss\@makefnmark}#1}
\RequirePackage[sort&compress]{natbib}
\newcommand\bibstyle@super{\bibpunct{[}{]}{,}{s}{,}{\textsuperscript{,}}}
\newcommand\bibstyle@numbers{\bibpunct{[}{]}{,}{n}{,}{,}}
\newcommand\bibstyle@authoryear{\bibpunct{(}{)}{;}{a}{,}{,}}
\def\tmp@numerical{numerical}
\def\tmp@authoryear{authoryear}
\newcommand\hnubibstyle[1]{%
  \def\tmp@gbt{#1}%
  \ifx\tmp@gbt\tmp@numerical%
    \bibliographystyle{hnunumerical}%
  \fi%
}

\if@hnu@numerical
  \if@hnu@super
    \citestyle{super}
  \else
    \citestyle{numbers}
  \fi
  \bibliographystyle{hnunumerical}%
\else
  \citestyle{authoryear}
  \bibliographystyle{hnuauthoryear}%
\fi

% 引用
\patchcmd\NAT@citexnum{%
  \@ifnum{\NAT@ctype=\z@}{%
    \if*#2*\else\NAT@cmt#2\fi
  }{}%
  \NAT@mbox{\NAT@@close}%
}{%
  \NAT@mbox{\NAT@@close}%
  \@ifnum{\NAT@ctype=\z@}{%
    \if*#2*\else\textsuperscript{#2}\fi
  }{}%
}{}{}
\renewcommand\NAT@citesuper[3]{\ifNAT@swa
\if*#2*\else#2\NAT@spacechar\fi
\unskip\kern\p@\textsuperscript{\NAT@@open#1\NAT@@close\if*#3*\else#3\fi}%
   \else #1\fi\endgroup}
\renewcommand\NAT@citenum%
    [3]{\ifNAT@swa\NAT@@open\if*#2*\else#2\NAT@spacechar\fi
        #1\NAT@@close\if*#3*\else\textsuperscript{#3}\fi\else#1\fi\endgroup}
\patchcmd{\NAT@citex}{%
  \if*#2*\else\NAT@cmt#2\fi
  \if\relax\NAT@date\relax\else\NAT@@close\fi
}{%
  \if\relax\NAT@date\relax\else\NAT@@close\fi
  \if*#2*\else\textsuperscript{#2}\fi
}{}{}
\renewcommand\NAT@cite%
    [3]{\ifNAT@swa\NAT@@open\if*#2*\else#2\NAT@spacechar\fi
        #1\NAT@@close\if*#3*\else\textsuperscript{#3}\fi\else#1\fi\endgroup}
\renewcommand\bibfont{\setfontsize{10.5bp}[20bp]}
\setlength{\bibsep}{0bp}
\setlength{\bibhang}{2\ccwd}
\renewcommand\@biblabel[1]{[#1]\hfill}
\urlstyle{same}
\g@addto@macro\UrlBreaks{%
  \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j%
  \do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t%
  \do\u\do\v\do\w\do\x\do\y\do\z%
  \do\A\do\B\do\C\do\D\do\E\do\F\do\G\do\H\do\I\do\J%
  \do\K\do\L\do\M\do\N\do\O\do\P\do\Q\do\R\do\S\do\T%
  \do\U\do\V\do\W\do\X\do\Y\do\Z%
  \do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9\do\0%
}
\renewcommand\bibsection{%
    \@mainmatterfalse
    \chapter{\bibname}%
    \@mainmattertrue
}

% 数学符号：
\RequirePackage{amsmath,amsthm}
\makeatletter
\g@addto@macro\normalsize{%
  \setlength\abovedisplayskip{16pt}
  \setlength\belowdisplayskip{16pt}
  \setlength\abovedisplayshortskip{16pt}
  \setlength\belowdisplayshortskip{16pt}
}
% \newcommand\eu{\mathrm{e}}
% \newcommand\iu{\mathrm{i}}
%\newcommand*{\diff}{\mathop{}\!\mathrm{d}}
%\DeclareMathOperator*{\argmax}{arg\,max}
%\DeclareMathOperator*{\argmin}{arg\,min}
% 数学定理：
% 以下定义数学定理环境默认风格为 hnuplain。
\newtheoremstyle{hnuplain}%
    {}{}%
    {}{2\ccwd}%
    {\bfseries}{}%
    {\ccwd}{}
\theoremstyle{hnuplain}
% 定义新的定理

\newcommand\hnu@assertionname{断言}
\newcommand\hnu@axiomname{公理}
\newcommand\hnu@corollaryname{推论}
\newcommand\hnu@definitionname{定义}
\newcommand\hnu@examplename{例}
\newcommand\hnu@lemmaname{引理}
\newcommand\hnu@proofname{证明}
\newcommand\hnu@propositionname{命题}
\newcommand\hnu@remarkname{注}
\newcommand\hnu@theoremname{定理}
\newtheorem{theorem}                {\hnu@theoremname}     [chapter]
\newtheorem{assertion}  [theorem]   {\hnu@assertionname}
\newtheorem{axiom}      [theorem]   {\hnu@axiomname}
\newtheorem{corollary}  [theorem]   {\hnu@corollaryname}
\newtheorem{lemma}      [theorem]   {\hnu@lemmaname}
\newtheorem{proposition}[theorem]   {\hnu@propositionname}
\newtheorem{definition}             {\hnu@definitionname}  [chapter]
\newtheorem{example}                {\hnu@examplename}     [chapter]
\newtheorem*{remark}                {\hnu@remarkname}
% \pkg{amsthm} 单独定义了 proof 环境，这里重新定义以满足格式要求。
% 原本模仿 \pkg{amsthm} 写成 |\item[\hskip\labelsep\hskip2\ccwd #1\hskip\ccwd]|，
% 但是却会多出一些间隙。
\renewenvironment{proof}[1][\proofname]{\par
    \pushQED{\qed}%
    \normalfont \topsep6\p@\@plus6\p@\relax
    \trivlist
        \item\relax\hskip2\ccwd
        \textbf{#1}
        \hskip\ccwd\ignorespaces
    }{%
    \popQED\endtrivlist\@endpefalse
}
\renewcommand\proofname\hnu@proofname

% 算法和代码：
% 算法环境
\RequirePackage[algoruled, algochapter, lined, linesnumbered]{algorithm2e}
\renewcommand{\algorithmcfname}{算法}
\RequirePackage{listings}
\lstset{
    basicstyle=\small\ttfamily,
    xleftmargin=2pc,
    xrightmargin=2pc,
    frame=single,
    columns=flexible,
    numbers=left,
}
\newcommand{\rememberlines}{\xdef\rememberedlines{\number\value{AlgoLine}}}
\newcommand{\resumenumbering}{\setcounter{AlgoLine}{\rememberedlines}}

\RequirePackage[skins]{tcolorbox}
\newtcolorbox{talgorithm}[2][]{%
  blanker,float=tbp,grow to left by=#2,grow to right by=#2,
  before upper={\begin{algorithm}[H]},
  after upper={\end{algorithm}},
  #1
}
