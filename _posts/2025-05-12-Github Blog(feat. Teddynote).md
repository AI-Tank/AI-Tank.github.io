
## 준비물
1. Github Desktop(desktop.github.com)
 
2. Typora : Markdown 언어 편집기(유료화됨) => Obsidian으로 대체

3. Visual Studio Code : 코드 편집기


## 이미지 업로드
1. Obsidian에 이미지 넣기 : 드래그 앤 드롭 방식
    기본적으로 마크 다운 언어(자체)는 이미지에 대한 편집기능을 지원하지 않는다. 
    => 플러그인을 설치하여 편하게 이미지를 편집한다.
   ![[Obsidian.jpg]]
   선택지 1 : Image Converter
	링크 : obsidian://show-plugin?id=image-converter
	우클릭을 통해 사진 편집, 드래그 방식으로 크기조정이 가능하다!
       
   선택지 2 : Image Toolkit
	링크 : obsidian://show-plugin?id=obsidian-image-toolkit
    이미지를 회전, 대칭, 색 대비 등의 기능은 보유하고 있으나 우클릭을 통한 상호작용 불가.

   일단 설치해보고 불편하면 지우자는 생각에 둘 다 설치하기로 결정했다.
<br><br><br><br>


2. 넣은 이미지를 github에 업로드하기.
  * Obisidian은 기본적으로 사진을 같은 폴더에 저장하고, 링크를 건다.
    => 즉 깃허브 블로그에 올릴 때에는 사진의 경로를 별도로 설정하여야 한다.
  * 사진파일의 경로를 Jekyll과 동일하게 \_images로 설정하고, 노트를 \_posts 폴더 안에 넣으면 자동으로 경로를 설정해주지 않을까? 라는 의문이 생겼다.
  * 아래와 같이 Attachment folder path를 "\_images"로 설정하고,
	  ![[image.png]]
 *  노트들의 주소도 "\_posts"로 설정했다.![[image-1.png]]
